"""
File: batched_agent_manager.py
Author: Matthew Allen

Description:
    A class to manage the multi-processed agents interacting with instances of the environment. This class is responsible
    for spawning and closing the individual processes, interacting with them through their respective pipes, and organizing
    the trajectories from each instance of the environment.
"""

import time
from typing import Union

import numpy as np
import torch

from rlgym_ppo.batched_agents import BatchedTrajectory
from rlgym_ppo.batched_agents.batched_agent import BatchedAgent

import ray

class BatchedAgentManager(object):
    def __init__(self, policy, min_inference_size=8, seed=123):
        self.policy = policy
        self.seed = seed
        self.batched_agents = []
        self.n_procs = 0
        self.current_obs = []
        self.average_reward = None
        self.cumulative_timesteps = 0
        self.min_inference_size = min_inference_size
        self.ep_rews = [] 
        self.trajectories = {}
        self.prev_time = 0

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
        states = []
        actions = []
        log_probs = []
        rewards = []
        next_states = []
        dones = []
        truncated = []

        n_collected = 0

        # Collect n timesteps.
        while n_collected < n:
            n_collected += self._step()

        # Organize our new timesteps into the appropriate lists.
        for trajectory in self.trajectories:
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

        self.cumulative_timesteps += n_collected
        t2 = time.perf_counter()

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
            n_collected,
            t2 - t1,
        )

    @torch.no_grad()
    def _step(self):
        """
        Send actions to environment processes based on current observations.
        """
        if self.current_obs is None:
            return

        if len(np.shape(self.current_obs)) == 2:
            self.current_obs = np.expand_dims(self.current_obs, 1)
        shape = np.shape(self.current_obs)

        actions, log_probs = self.policy.get_action(np.stack(self.current_obs, axis=1))
        actions = actions.view((shape[0], shape[1], -1)).numpy()
        log_probs = log_probs.view((shape[0], shape[1], -1)).numpy()

        results = ray.get([agent.step.remote(actions[i]) for i, agent in enumerate(self.batched_agents)])
        prev_obs = self.current_obs
        self.current_obs = []
        
        n_collected = 0

        for i, trajectory in enumerate(self.trajectories):
            action = actions[i]
            log_prob = log_probs[i]
            state = np.asarray(prev_obs[i])
            next_state, reward, done = results[i]
            
            if type(reward) in (list, tuple, np.ndarray):
                n_collected += len(reward)
                for j in range(len(reward)):
                    if j >= len(self.ep_rews[i]):
                        self.ep_rews[i].append(reward[j])
                    else:
                        self.ep_rews[i][j] += reward[j]
            else:
                n_collected += 1
                self.ep_rews[i][0] += reward
            
            if done:
                if self.average_reward is None:
                    self.average_reward = self.ep_rews[i][0]
                else:
                    for ep_rew in self.ep_rews[i]:
                        self.average_reward = (
                            self.average_reward * 0.9 + ep_rew * 0.1
                        )
                
                self.ep_rews[i] = [0]
                
            trajectory.action = action
            trajectory.log_prob = log_prob
            trajectory.state = state
            trajectory.reward = reward
            trajectory.next_state = next_state
            trajectory.done = done
            trajectory.update()
            self.current_obs.append(next_state)

        return n_collected

    def _get_initial_states(self):
        """
        Retrieve initial states from environment processes.
        :return: None.
        """

        self.current_obs = ray.get([agent.reset.remote() for agent in self.batched_agents])

    def _get_env_shapes(self):
        """
        Retrieve environment observation and action space shapes from one of the connected environment processes.
        :return: A tuple containing observation shape, action shape, and action space type.
        """

        print("Requesting env shapes...")
        return ray.get(self.batched_agents[0].get_env_shapes.remote())

    def init_processes(
        self,
        n_processes,
        build_env_fn,
        spawn_delay=None,
        render=False,
        render_delay: Union[float, None] = None,
    ):
        """
        Initialize and spawn environment processes.

        :param n_processes: Number of processes to spawn.
        :param build_env_fn: A function to build the environment for each process.
        :param spawn_delay: Delay between spawning processes. Defaults to None.
        :param render: Whether an environment should be rendered while collecting timesteps.
        :param render_delay: A period in seconds to delay a process between frames while rendering.
        :return: A tuple containing observation shape, action shape, and action space type.
        """
        
        self.n_procs = n_processes

        self.batched_agents = []
        for i in range(n_processes):
            render_this_proc = i == 0 and render
            self.batched_agents.append(
                BatchedAgent.remote(
                build_env_fn,
                self.seed + i,
                render_this_proc,
                render_delay
            ))
            if spawn_delay is not None:
                time.sleep(spawn_delay)

        self.ep_rews = [[0] for _ in range(n_processes)]
        self.trajectories = [BatchedTrajectory() for _ in range(n_processes)]
        self._get_initial_states()
        return self._get_env_shapes()

    def cleanup(self):
        ray.shutdown()
