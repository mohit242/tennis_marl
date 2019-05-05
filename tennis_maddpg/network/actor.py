import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class DDPGActor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_units=(128, 128, 64), gate=nn.ReLU()):
        """ Network class for PPO actor network

        Args:
            state_dim: dimension of states/observations
            action_dim: dimension of action vector
            hidden_units : list of number of hidden layer neurons
            gate: activation gate
        """
        super().__init__()
        dims = (state_dim, ) + hidden_units + (action_dim, )
        linear_func = lambda a, b: nn.Linear(a, b)
        act_func = lambda a, b: gate
        layers = [f(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:]) for f in (linear_func, act_func)]
        layers = layers[:-1]
        self.network = nn.Sequential(*layers)
        self.network.apply(self.init_layer)
        self.network[-1].weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        # state /= 100.0
        action = F.tanh(self.network(state))
        return action

    def init_layer(self, layer):
        if isinstance(layer, nn.Linear):
            fan_in = layer.weight.data.size()[0]
            lim = 1. / np.sqrt(fan_in)
            layer.weight.data.uniform_(-lim, lim)