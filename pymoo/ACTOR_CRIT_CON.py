import torch
import torch.nn as nn
from torch.distributions import Normal
import torch.nn.functional as F

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        self.shared = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )

        self.mu = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

        self.value = nn.Linear(128, 1)

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
