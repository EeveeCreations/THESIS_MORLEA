import torch
import torch.nn as nn
from torch.distributions import Normal




###### BASE NON CHANGING  PARAMTERS ##############################
LAYERS_NN = 4
ACTIVATION_F_NN = nn.ReLU
FILTERS_IN_NN = 128
FILTERS_OUT_NN = 128
FILTERS_MID_NN = 128

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, FILTERS_OUT_NN),
            ACTIVATION_F_NN(),
            nn.Linear(FILTERS_IN_NN, FILTERS_OUT_NN),
            ACTIVATION_F_NN(),
        )

        self.mu = nn.Linear(FILTERS_IN_NN, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.value = nn.Linear(FILTERS_IN_NN
                               , 1)

    def forward(self, state):
        x = self.shared(state)

        mu = torch.tanh(self.mu(x))  # bound mean
        std = torch.exp(self.log_std)

        value = self.value(x)

        return mu, std, value


    def act(self, state):
        mu, std, value = self.forward(state)
        dist = Normal(mu,std)
        action = dist.sample()
        return action, dist.log_prob(action), value
