import numpy as np

#Should start working as  a baseline

class RandomAgent(n_actions):
    """Randomly picks parameters each generation — our baseline."""
    def __init__(self):
        self.episode_rewards = []

    def select_action(self, state):
        return np.random.randint(n_actions)

    def update(self, *args): pass

    def end_episode(self, total_reward):
        self.episode_rewards.append(total_reward)
