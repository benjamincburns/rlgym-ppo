"""
File: multi_stepping_agent_manager.py
Authors: Matthew Allen, Ben Burns

Description:
    A class to manage the vectorized agents interacting with instances of the environment. This class is responsible
    for spawning and closing the individual threads, interacting with them through their respective queues, and organizing
    the trajectories from each instance of the environment.
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from typing import Union

import gym

import numpy as np
import torch

from rlgym_ppo.batched_agents import BatchedTrajectory, comm_consts
from rlgym_ppo.batched_agents.multi_stepping_agent import MultiSteppingAgent
from rlgym_ppo.util import WelfordRunningStat

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterator, *args, **kwargs):
        return iterator


class MultiSteppingAgentManager(object):
    def __init__(
        self,
        policy,
        min_inference_size=8,
        seed=123,
        standardize_obs=True,
        steps_per_obs_stats_increment=5,
    ):
        
        """
        Collects timesteps from the policies managed by this class.
        
        :param policy: The policy to use for action selection.
        :param min_inference_size: IGNORED - kept only for interface compatibility
        :param seed: Seed for environment and action space randomization.
        :param standardize_obs: Whether to standardize observations.
        :param steps_per_obs_stats_increment: Number of timesteps to collect before updating observation statistics.
        """
        self.policy = policy
        self.seed = seed

        self.average_reward = None
        self.cumulative_timesteps = 0
        
        # ignored, but kept for signature compatibility
        self.min_inference_size = min_inference_size

        self.standardize_obs = standardize_obs
        self.steps_per_obs_stats_increment = steps_per_obs_stats_increment
        self.steps_since_obs_stats_update = 0
        
        # TODO - handle obs standardization
        self.obs_stats = None

        self.ep_rews = {}
        self.trajectory_map = {}
        self.completed_trajectories = []

        self.pool = None
        self.n_threads = 0
        self.agents = []

    def collect_timesteps(self, n):
        """
        Collect a specified number of timesteps from the environment.

        :param n: Number of timesteps to collect.
        :return: A tuple containing the collected data arrays and additional information.
                - states (np.ndarray): Array of states.
                - actions (np.ndarray): Array of actions.
                - log_probs (np.ndarray): Array of log probabilities of actions.
                - rewards (np.ndarray): Array of rewards.
                - next_states (np.ndarray): Array of next states.
                - dones (np.ndarray): Array of done flags.
                - truncated (np.ndarray): Array of truncated flags.
                - n_collected (int): Number of timesteps collected.
                - elapsed_time (float): Time taken to collect the timesteps.
        """

        t1 = time.perf_counter()

        # split steps to be collected across the agent pool
        n_agents = len(self.agents)
        steps_to_collect = [n // n_agents] * n_agents

        # handle the case where the number of steps to collect doesn't divide
        # evenly by the number of agents
        i = 0
        while sum(steps_to_collect) < n:
            steps_to_collect[i] += 1
            i += 1
        
        collected_metrics = []
        
        # collect the steps asynchronously using the ThreadPoolExecutor
        futures = {self.pool.submit(agent.collect_timesteps, steps_to_collect[i], self.policy): i for i, agent in enumerate(self.agents) }
        for future in as_completed(futures):
            agent_id = futures[future]
            timesteps = future.result()

            for timestep in timesteps:
                state, action, log_prob, reward, next_state, done, info = timestep
                self.trajectory_map[agent_id].add_timestep(state, action, log_prob, reward, next_state, done)

                # TODO: increment reward stats
                # TODO: handle info
                if "metrics" in info:
                    collected_metrics.append(info["metrics"])
                if done:
                    self.completed_trajectories.append(self. trajectory_map[agent_id])
                    self.trajectory_map[agent_id] = BatchedTrajectory()
        
        states = []
        actions = []
        log_probs = []
        rewards = []
        next_states = []
        dones = []
        truncated = []

        # Organize our new timesteps into the appropriate lists.
        for i, trajectory in self.trajectory_map.items():
            self.completed_trajectories.append(trajectory)
            self.trajectory_map[i] = BatchedTrajectory()

        for trajectory in self.completed_trajectories:
            trajectories = trajectory.get_all()
            if len(trajectories) == 0:
                continue

            for traj in trajectories:
                (
                    trajectory_states,
                    trajectory_actions,
                    trajectory_log_probs,
                    trajectory_rewards,
                    trajectory_next_states,
                    trajectory_dones,
                ) = traj
                trajectory_truncated = [0 for _ in range(len(trajectory_dones))]
                trajectory_truncated[-1] = 1 if trajectory_dones[-1] == 0 else 0
                states += trajectory_states
                actions += trajectory_actions
                log_probs += trajectory_log_probs
                rewards += trajectory_rewards
                next_states += trajectory_next_states
                dones += trajectory_dones
                truncated += trajectory_truncated

        self.cumulative_timesteps += n
        t2 = time.perf_counter()
        self.completed_trajectories = []

        return (
            (
                np.asarray(states),
                np.asarray(actions),
                np.asarray(log_probs),
                np.asarray(rewards),
                np.asarray(next_states),
                np.asarray(dones),
                np.asarray(truncated),
            ),
            collected_metrics,
            n,
            t2 - t1,
        )


    def init_environments(
        self,
        n_envs,
        n_threads,
        build_env_fn,
        collect_metrics_fn=None,
        spawn_delay=None,
        render=False,
        render_delay: Union[float, None] = None,
    ):
        """
        Initialize and spawn environments.
        :param n_envs: Number of vectorized, multi-stepping environments to spawn.
        :param build_env_fn: A function to build the environment for each process.
        :param collect_metrics_fn: A user-defined function that the environment processes will use to collect metrics
               about the environment at each timestep.
        :param spawn_delay: Delay between spawning environment instances. Defaults to None.
        :param render: Whether an environment should be rendered while collecting timesteps.
        :param render_delay: A period in seconds to delay a process between frames while rendering.
        :return: A tuple containing observation shape, action shape, and action space type.
        """
        
        # the environment is vectorized, meaning each env here actually
        # represents multiple parallel environments under the hood.
        
        # the threads that we're spawning here handle batched inference for
        # each batch of agents in each environment.
        
        # assuming that we're using CPU for inference, we only want one management
        # thread, as the CPU will be used for both inference and stepping the
        # environment.
        n_threads = 1

        n_gpus = 0

        # if we're using GPU, we want two management threads per GPU. This way,
        # one thread will be stepping the vectorized envs while the other is
        # performing inference on the GPU.
        if torch.cuda.is_available():
            n_gpus = torch.cuda.device_count()
            n_threads = n_gpus * 2
        
        if n_threads > n_envs:
            # I don't know why you'd use this for this case, but edge cases need
            # to be addressed, I guess?
            n_threads = n_envs
        
        self.pool = ThreadPoolExecutor(max_workers=n_threads)
        
        self.n_threads = n_threads

        # Spawn threads
        for i in tqdm(range(n_envs)):
            render_this_proc = i == 0 and render
            
            device = "cpu"

            if n_gpus > 0:
                device = f"cuda:{str(i % n_gpus)}"
            
            self.agents.append(MultiSteppingAgent(build_env_fn=build_env_fn,
                                       collect_metrics_fn=collect_metrics_fn,
                                       render=render_this_proc,
                                       render_delay=render_delay,
                                       seed=self.seed + i,
                                       device=device))

            self.ep_rews[i] = [0]
            self.trajectory_map[i] = BatchedTrajectory()

        env = self.agents[0]
        obs_shape = np.prod(env.observation_space.shape)

        if hasattr(env.action_space, "n"):
            action_shape = env.action_space.n
        else:
            action_shape = np.prod(env.action_space.shape)

        t = type(env.action_space)
        action_space_type = 0  # "discrete"
        if t == gym.spaces.multi_discrete.MultiDiscrete:
            action_space_type = 1  # "multi-discrete"
        elif t == gym.spaces.box.Box:
            action_space_type = 2  # "continuous"
        
        return int(obs_shape), int(action_shape), action_space_type

    def cleanup(self):
        """
        Clean up resources and terminate processes.
        """
        import traceback
        try:
            if self.pool is not None:
                self.pool.shutdown(wait=True, cancel_futures=True)
        except Exception:
            print("Unable to shutdown ThreadPoolExecutor")
            traceback.print_exc()
