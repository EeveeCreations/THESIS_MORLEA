import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
import copy

class DQNAgent:
    """
    DQN with experience replay and a target network.
    Handles continuous state directly — no discretization needed.
    """

    def __init__(self, lr=1e-3, gamma=0.95, epsilon=1.0, epsilon_decay=0.97,
                 batch_size=64, target_update_freq=5):
        self.gamma            = gamma
        self.epsilon          = epsilon
        self.epsilon_min      = 0.05
        self.epsilon_decay    = epsilon_decay
        self.batch_size       = batch_size
        self.target_update_freq = target_update_freq
        self.train_step       = 0

        self.policy_net = DQNNetwork()
        self.target_net = DQNNetwork()
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn    = nn.SmoothL1Loss()    # Huber loss — more stable than MSE
        self.buffer     = ReplayBuffer()
        self.episode_rewards = []
        self.losses     = []

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0)
            return int(self.policy_net(s).argmax().item())

    def update(self, state, action, reward, next_state, done):
        self.buffer.push(state, action, reward, next_state, float(done))
        if len(self.buffer) < self.batch_size:
            return

        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        # Current Q values
        q_values = self.policy_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q values (using frozen target network)
        with torch.no_grad():
            max_next_q = self.target_net(ns).max(1)[0]
            q_targets  = r + self.gamma * max_next_q * (1 - d)

        loss = self.loss_fn(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.losses.append(loss.item())

        # Periodically sync target network
        self.train_step += 1
        if self.train_step % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

    def end_episode(self, total_reward):
        self.episode_rewards.append(total_reward)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


