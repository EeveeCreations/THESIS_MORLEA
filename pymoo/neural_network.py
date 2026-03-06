import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from torch.distributions import Categorical

####### NN  Paramaters ###############################
LAYERS_NN = 4
ACTIVATION_F_NN = nn.ReLU
FILTERS_IN_NN = 128
FILTERS_OUT_NN = 128
FILTERS_MID_NN = 256




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


class DQNNetwork(nn.Module):
    """Small MLP: state → Q-values for each action."""
    def __init__(self, state_dimensions, n_actions):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dimensions, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience replay: stores (s, a, r, s', done) tuples."""
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(ns)),
                torch.FloatTensor(d))

    def __len__(self):
        return len(self.buffer)
