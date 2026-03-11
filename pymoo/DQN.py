from collections import deque
import numpy as np
import random
import torch
import torch.nn as nn



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

class TargetNetwork(DQNNetwork):
    def __init__(self, state_dimensions, n_actions):
        super(TargetNetwork, self).__init__(state_dimensions, n_actions)



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
