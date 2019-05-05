import torch
import numpy as np
import random
from collections import deque, namedtuple


def combine_agent_tensors(x):
    y = torch.cat((x[:, 0], x[:, 1]), -1)
    return y


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

    def sample(self, priority=False):
        """Randomly sample a batch from memory"""

        if priority == True:
            probs = np.array([max(e.reward[0], e.reward[1]) for e in self.memory if e is not None])
            probs = np.exp(probs)
            probs = probs / probs.sum()
            idx = np.random.choice(len(self.memory), self.batch_size, p=probs)
            experiences = [self.memory[i] for i in idx]
        else:
            experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.array([e.state for e in experiences if e is not None])).float()
        actions = torch.from_numpy(np.array([e.action for e in experiences if e is not None])).long()
        rewards = torch.from_numpy(np.array([e.reward for e in experiences if e is not None])).float()
        next_states = torch.from_numpy(np.array([e.next_state for e in experiences if e is not None])).float()
        dones = torch.from_numpy(np.array([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.memory)
