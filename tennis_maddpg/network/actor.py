import torch
import torch.nn as nn
import torch.nn.functional as F

class DDPGActor(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_units=(128, 64, 64), gate=nn.ELU()):
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

    def forward(self, state):
        state = torch.Tensor(state)
        action = torch.clamp(F.tanh(self.network(state)), -1.0, 1.0)
        return action

    def init_layer(self, layer):
        if isinstance(layer, nn.Linear):
            torch.nn.init.kaiming_uniform_(layer.weight)
            torch.nn.init.constant_(layer.bias, 0.0)