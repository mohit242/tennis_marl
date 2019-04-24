import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPGCritic(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_units=(128, 128, 64), gate=nn.ELU()):
        """ Network class for PPO actor network

        Args:
            state_dim: dimension of states/observations
            hidden_units: list of number of hidden layer neurons
            gate: activation gate
        """
        super().__init__()
        self.fcs1 = nn.Linear(state_dim, hidden_units[0])
        dims = (hidden_units[0] + action_dim, ) + hidden_units[1:] + (1, )
        linear_func = lambda a, b: nn.Linear(a, b)
        act_func = lambda a, b: gate
        layers = [f(dim_in, dim_out) for dim_in, dim_out in zip(dims[:-1], dims[1:]) for f in (linear_func, act_func)]
        layers = layers[:-1]
        self.network = nn.Sequential(*layers)
        self.network.apply(self.init_layer)
        self.fcs1.apply(self.init_layer)

    def forward(self, state, action):
        state = torch.Tensor(state)
        action = torch.Tensor(action)
        xs = F.elu(self.fcs1(state))
        x = torch.cat((xs, action), dim=-1)
        qval = self.network(x)
        return qval

    def init_layer(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_uniform_(layer.weight)
            torch.nn.init.constant_(layer.bias, 1.0)