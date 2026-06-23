import torch
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem

from AC import ActorCritic
from actorcritic import *
from ea_environment import *
# from pymoo.continues.parameters.MOEA_RL import USED_PROBLEM_NAME, crossover_probability, mutation_probability, USED_SEED

# from continues.paramters.MOEA_RL import USED_PROBLEM, USED_ALGORITHM
### PROBLEM  / ALGORITHEM USED
# ic  Soltion functiom

USED_SEED = 42
### Main   Dynamic Parameters N#############################################################################
USED_PROBLEM_NAME= "zdt4"

### EVO ALGO NSGAII ##################
crossover_probability = 0.9
mutation_probability = 0.9
max_generations =20

USED_PROBLEM = get_problem(USED_PROBLEM_NAME)
USED_ALGORITHM = NSGA2(
    crossover=SBX(eta=15, prob=crossover_probability),
    mutation=PM(eta=20, prob=mutation_probability),
    pop_size=1000
)
USED_ALGORITHM.setup(USED_PROBLEM, seed=USED_SEED)






###### MODIFYIANBLE   PARAMTERS PPO ##############################

GAMMA = 0.99
LAMBDA= 0.95
CLIP= 0.3
LEARNING_RATE= 2e-5
EPOCHS = 10
ENTHROPHY_COUNT = 0.1
ACTOR_LOSS = 0.8


class PPO(ActorCritic):
    def __init__(self, state_dim, action_dim):
        super().__init__(state_dim, action_dim)
        self.gamma = GAMMA
        self.lam = LAMBDA
        self.clip_eps = CLIP
        self.lr = LEARNING_RATE
        self.epochs = EPOCHS
        self.entropy_count = ENTHROPHY_COUNT
        self.actor_loss = ACTOR_LOSS

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
            #0,5 =   somethign    self.gamma was 0.1
            loss = actor_loss + self.actor_loss * critic_loss - self.entropy_count * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()


### TRain loop
def train(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim)

    max_episodes = 200

    for episode in range(max_episodes):

        state,_ = env.reset()  # Might add info later

        states = []
        actions = []
        rewards = []
        log_probs = []
        values = []
        dones = []

        done = False

        while not done:
            print(state)
            state_tensor = torch.FloatTensor(state)
            action, log_prob, value = agent.model.act(state_tensor)

            next_state, reward, terminated, truncated = env.step(action.item())
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
    env = EAEnv(USED_ALGORITHM, USED_PROBLEM)
    train(env)
