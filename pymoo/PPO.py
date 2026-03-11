import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from AC import ActorCritic
from neural_network import *
from ea_environment import *
from MOEA_RL import USED_PROBLEM, USED_ALGORITHEM

###### MODIFYIANBLE   PARAMTERS PPO ##############################

GAMMA = 0.99
LAMBDA= 0.95
CLIP= 2e-2
LEARNING_RATE= 2e-4
EPOCHS = 10

class PPO(ActorCritic):
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA
        self.lam = LAMBDA
        self.clip_eps = CLIP
        self.lr = LEARNING_RATE
        self.epochs = EPOCHS

        self.model = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1-dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1-dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def update(self, states, actions, log_probs_old, returns, advantages):

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        log_probs_old = torch.FloatTensor(log_probs_old)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        for _ in range(self.epochs):

            logits, values = self.model(states)
            dist = Categorical(logits=logits)
            log_probs = dist.log_prob(actions)

            ratio = torch.exp(log_probs - log_probs_old)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)

            entropy = dist.entropy().mean()

            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


### TRain loop
def train(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = PPO(state_dim, action_dim)

    max_episodes = 500

    for episode in range(max_episodes):

        state, _ = env.reset()

        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []

        done = False

        while not done:

            state_tensor = torch.FloatTensor(state)
            action, log_prob, value = agent.model.act(state_tensor)

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            states.append(state)
            actions.append(action.item())
            rewards.append(reward)
            log_probs.append(log_prob.item())
            values.append(value.item())
            dones.append(done)

            state = next_state

        advantages = agent.compute_gae(rewards, values, dones)
        returns = [a + v for a, v in zip(advantages, values)]

        agent.update(states, actions, log_probs, returns, advantages)

        total_reward = sum(rewards)

        print(f"Episode {episode} | Reward: {total_reward}")

    env.close()


if __name__ == "__main__":
    env = EAEnv(USED_ALGORITHEM, USED_PROBLEM)
    train(env)
