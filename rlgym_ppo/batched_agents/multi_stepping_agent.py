import numpy as np
import time
import torch

class MultiSteppingAgent(object):
    def __init__(self, build_env_fn, metrics_encoding_function, seed, render, render_delay, device):
        self.seed = seed
        self.render = render
        self.render_delay = render_delay
        
        self.metrics_encoding_function = metrics_encoding_function
        self.env = build_env_fn()

        # seed action space
        self.env.action_space.seed(seed)

        reset_state = self.env.reset()

        if type(reset_state) != np.ndarray:
            reset_state = np.asarray(reset_state, dtype=np.float32)
        elif reset_state.dtype != np.float32:
            reset_state = reset_state.astype(np.float32)

        self.prev_obs = reset_state
        
        self.device = device
    
    @torch.no_grad()
    def collect_timesteps(self, n_steps, policy):
        """
        Function to interact with an environment

        :param n_steps: The number of timesteps to collect
        :param policy: The policy to use for action selection
        """

        obs_shape = self.prev_obs.shape
        if len(obs_shape) != 2:
            throw = f"Observation shape must be 2D, got {len(obs_shape)}D"

        n_agents = self.prev_obs.shape[0]
        
        timesteps = []

        # Primary interaction loop.
        try:
            collected_steps = 0
            while collected_steps < n_steps:

                inference_batch = np.concatenate(self.prev_obs, axis=0)
                actions, log_probs_vec = policy.get_action(inference_batch)
                actions: np.ndarray = actions.numpy().astype(np.float32)
                
                # reshape in to (n_agents, action_dim)
                actions.reshape((n_agents, -1))

                obs_vec, rew_vec, done_vec, info_vec = self.env.step(actions)

                # we collect n_agents steps per call to the environment step function
                collected_steps += n_agents
                
                if type(obs_vec) != np.ndarray:
                    obs_vec = np.asarray(obs_vec, dtype=np.float32)

                elif obs_vec.dtype != np.float32:
                    obs_vec = obs_vec.astype(np.float32)

                obs_vec.reshape((n_agents, -1))

                for i in range(n_agents):
                    if self.metrics_encoding_function is not None:
                        info_vec[i]["metrics"] = self.metrics_encoding_function(info_vec[i]["state"])
                
                    timesteps.append((self.prev_obs[i], actions[i], log_probs_vec[i], rew_vec[i], obs_vec[i], done_vec[i], info_vec[i]))
                
                self.prev_obs = obs_vec

                if self.render:
                    self.env.render()
                    if self.render_delay is not None:
                        time.sleep(self.render_delay)

            return timesteps

        except Exception:
            import traceback

            print("ERROR IN MULTISTEP AGENT LOOP")
            traceback.print_exc()
