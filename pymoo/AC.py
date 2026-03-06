import torch.nn as nn
import numpy as np
from torch.distributions import Categorical


####### NN  Paramaters ###############################
LAYERS_NN = 4
ACTIVATION_F_NN = nn.ReLU
FILTERS_IN_NN = 128
FILTERS_OUT_NN = 128
FILTERS_MID_NN = 256

### PPO Parameters
GAMMA = 0.99
LAMBDA = 0.99
GEA =0.99
LR = 2e-4 # My favoritee
CLIP = 1e-2


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()

        model = nn.Sequential()
        model.append(nn.Linear(state_dim, FILTERS_OUT_NN))
        for layer in range(LAYERS_NN):
               model.append(ACTIVATION_F_NN())
               model.append(nn.Linear(FILTERS_MID_NN,FILTERS_MID_NN))

        self.shared  = model
        self.actor = nn.Linear(FILTERS_OUT_NN, action_dim)
        self.critic = nn.Linear(FILTERS_MID_NN, 1)

    def forward(self, x):
        x = self.shared(x)
        return self.actor(x), self.critic(x)

    def act(self, state):
        logits, value = self.forward(state)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action, dist.log_prob(action), value
