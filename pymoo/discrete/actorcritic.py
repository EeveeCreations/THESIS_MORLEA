import torch.nn as nn
from collections import deque, namedtuple
from torch.distributions import Categorical

####### NN  Paramaters ###############################
LAYERS_NN = 4
ACTIVATION_F_NN = nn.ReLU
FILTERS_IN_NN = 128
FILTERS_OUT_NN = 128
FILTERS_MID_NN = 128

class ActorCritic(nn.Module):
    def __init__(self, tate_dimensions, action_dim):
        super().__init__()

        model = nn.Sequential()
        model.append(nn.Linear(tate_dimensions, FILTERS_OUT_NN))
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

