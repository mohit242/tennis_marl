import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPGCritic(nn.Module):

    def __init__(self, state_dim, hidden_units=(64, 64), gate=nn.ELU()):
        """ Network class for PPO actor network

        Args:
            state_dim: dimension of states/observations
            hidden_units: list of number of hidden layer neurons
            gate: activation gate
        """
        super().__init__()
        dims = (state_dim, ) + hidden_units + (1, )
        linear_func = lambda a, b: nn.Linear(a, b)
        act_func = lambda a, b: gate
        layers = [f(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:]) for f in (linear_func, act_func)]
        layers = layers[:-1]
        self.network = nn.Sequential(*layers)

    def forward(self, state):
        state = torch.Tensor(state)
        qval = self.network(state)
        return qval
