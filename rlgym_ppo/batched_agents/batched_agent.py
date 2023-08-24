from typing import Union
import ray
import time
import gym

@ray.remote
class BatchedAgent(object):
    def __init__(self, idx, build_env_fn, seed, render, render_delay: Union[float, None]):
        """
        :param seed: Seed for environment and action space randomization.
        :param render: Whether the environment will be rendered every timestep.
        :param render_delay: Amount of time in seconds to delay between steps while rendering.
        """
        self.idx = idx
        self.env = build_env_fn()
        self.seed = seed
        self.render=render
        self.render_delay=render_delay

        # Seed everything.
        self.env.action_space.seed(seed)
    
    def step(self, actions, action_index):
        action = actions[action_index]
        obs, rew, done, _ = self.env.step(action)
        if done:
            obs = self.env.reset()

        if self.render:
            self.env.render()
            if self.render_delay is not None:
                time.sleep(self.render_delay)

        return self.idx, obs, rew, done

    
    def reset(self):
        return self.env.reset()
    
    def get_env_shapes(self):
        t = type(self.env.action_space)
        action_space_type = "discrete"
        if t == gym.spaces.multi_discrete.MultiDiscrete:
            action_space_type = "multi-discrete"
        elif t == gym.spaces.box.Box:
            action_space_type = "continuous"

        if hasattr(self.env.action_space, "n"):
            n_acts = self.env.action_space.n
        else:
            n_acts = self.env.action_space.shape

        return self.env.observation_space.shape, n_acts, action_space_type
