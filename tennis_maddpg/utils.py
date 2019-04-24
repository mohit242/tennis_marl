import torch
import numpy as np
import random
from collections import deque, namedtuple


def process_actions(action):
    dist = torch.distributions.Normal(0, 0.15)
    noise = dist.sample(action.size())
    noise = torch.clamp(noise, -0.1, 0.1)
    action = torch.clamp(action + noise, -1, 1)
    return action


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed=0):
        """Fixed-size buffer to store experience tuples.

        Args:
            action_size (int): dimension of action space
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
            seed (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        self.Experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory

        Args:
            state:
            action:
            reward:
            next_state:
            done:

        Returns:

        """
        e = self.Experience(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch from memory"""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
