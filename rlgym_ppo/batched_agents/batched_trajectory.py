"""
File: batched_trajectory.py
Author: Matthew Allen

Description:
    A class to maintain timesteps from batched agents in synchronized sequences.
"""

import numpy as np


class BatchedTrajectory(object):
    def __init__(self):
        self.state = None
        self.action = None
        self.log_prob = None
        self.reward = None
        self.next_state = None
        self.done = None
        self.complete_timesteps = []

    def update(self):
        """
        Function to check if the current timestep data is ready to be appended to the sequence we are tracking.
        :return: None.
        """

        # If every class attribute is populated
        if self.state is not None and\
           self.action is not None and\
           self.log_prob is not None and\
           self.reward is not None and\
           self.next_state is not None and\
           self.done is not None:

            # If there is only a single agent in the match, create a list out of the scalar values.
            if type(self.reward) not in (list, tuple, np.ndarray):
                self.action = [self.action]
                self.log_prob = [self.log_prob]
                self.reward = [self.reward]
                self.done = self.done

            # Append timestep data to our sequence and reset all class attributes.
            self.complete_timesteps.append((self.state, self.action, self.log_prob, self.reward, self.next_state, self.done))
            self.state = None
            self.action = None
            self.log_prob = None
            self.reward = None
            self.next_state = None
            self.done = None
        
    def get_all(self):
        """
        Function to retrieve and erase all timestep sequences tracked by this object.
        :return: List of completed sequences.
        """

        if len(self.complete_timesteps) == 0:
            return []

        trajectories = []

        # We are tracking data from a single match, not agent, so we will have n_agents number of trajectories in our list.
        n_trajectories = self.complete_timesteps[0][0].shape[0]

        # For each trajectory we are tracking.
        for i in range(n_trajectories):
            states = []
            rewards = []
            actions = []
            log_probs = []
            next_states = []
            dones = []

            # Acquire all the timesteps from the current trajectory and append them to our lists.
            for timestep in self.complete_timesteps:
                state, action, log_prob, reward, next_state, done = timestep
                states.append(state[i])
                actions.append(action[i])
                log_probs.append(log_prob[i])
                next_states.append(next_state[i])
                dones.append(done)
                rewards.append(reward[i])

            trajectories.append([states, actions, log_probs, rewards, next_states, dones])

        # Reset our trajectory buffer and return all trajectories.
        self.complete_timesteps = []
        return trajectories
