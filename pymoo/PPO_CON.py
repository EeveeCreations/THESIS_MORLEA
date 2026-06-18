import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
from jinja2.lexer import float_re
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.problems import get_problem
from torch.distributions import Normal
from torch.utils.tensorboard import SummaryWriter

from ACTOR_CRIT_CON import ActorCritic
# from MOEA_RL import USED_PROBLEM, USED_ALGORITHM

from EA_ENV_CON import EAEnv

USED_SEED = 42
### Main   Dynamic Parameters N#############################################################################
USED_PROBLEM_NAME= "zdt4"

### EVO ALGO NSGAII ##################
crossover_probability = 0.7 #(0,1.0)
mutation_probability = 0.02 #(0, 1.0)
max_generations = 20

USED_PROBLEM = get_problem(USED_PROBLEM_NAME)
USED_ALGORITHM = NSGA2(
    crossover=SBX(eta=15, prob=crossover_probability),
    mutation=PM(eta=20, prob=mutation_probability),
    pop_size=1000
)
USED_ALGORITHM.setup(USED_PROBLEM, seed=USED_SEED)

###### MODIFYIANBLE   PARAMTERS PPO ##############################
GAMMA = 0.97
LAMBDA= 0.98
CLIP= 0.005
LEARNING_RATE= 2e-4
EPOCHS = 20
ENTHROPHY_COUNT = 0.1
ACTOR_LOSS = 0.8
MODEL = ActorCritic
OPTIMIZER= optim.Adam

class PPO:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA
        self.lam = LAMBDA
        self.clip_eps = CLIP
        self.epochs = EPOCHS
        self.lr = LEARNING_RATE

        self.model = MODEL(state_dim, action_dim)
        self.optimizer = OPTIMIZER(self.model.parameters(), lr=self.lr)

    def act(self, state):
        state = torch.FloatTensor(state)

        mu, std, value = self.model(state)

        dist = Normal(mu, std)
        action = dist.sample()

        log_prob = dist.log_prob(action).sum(dim=-1)

        return action.detach().numpy(), log_prob.detach(), value.detach()

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)

        return advantages

    def update(self, states, actions, old_log_probs, returns, advantages):
        print( type(states), type(actions), type(old_log_probs), type(returns), type(advantages))
        states = torch.FloatTensor(np.array(states))
        actions = torch.FloatTensor(np.array(actions))
        old_log_probs = torch.FloatTensor(old_log_probs)
        returns = torch.FloatTensor(returns)
        advantages = torch.FloatTensor(advantages)

        for _ in range(self.epochs):

            mu, std, values = self.model(states)

            dist = Normal(mu, std)
            log_probs = dist.log_prob(actions).sum(dim=-1)

            ratio = torch.exp(log_probs - old_log_probs)

            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * advantages

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values.squeeze(), returns)

            loss = actor_loss + 0.5 * critic_loss

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()



    ### TRain loop
def train(env):
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    agent = PPO(state_dim, action_dim)

    max_episodes = 20

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
            action, log_prob, value = agent.act(state_tensor) # Model.act or   ppo.act

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            states.append(state.copy())
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value.item())
            dones.append(done)

            state = next_state

        advantages = agent.compute_gae(rewards, values, dones)
        returns = [a + v for a, v in zip(advantages, values)]

        agent.update(states, actions, log_probs, returns, advantages)

        total_reward = sum(rewards)
        ## SAFE TO   CSV
        writer.add_scalar(
            "Reward/TotalReward",
            total_reward,
            episode
        )

        writer.add_scalar(
            "Advantage/Mean",
            np.mean(advantages),
            episode
        )

        writer.add_scalar(
            "Return/Mean",
            np.mean(returns),
            episode
        )
        # with open(csv_file, "a", newline="") as f:
        #     writer = csv.writer(f)
        #     writer.writerow([
        #         episode,
        #         total_reward,
        #         np.mean(advantages),
        #         np.mean(returns)
        #     ])


        print(f"Episode {episode} | Reward: {total_reward}")

    env.close()
    return agent


if __name__ == "__main__":
    ######## Save the   ALgorithem
    writer = SummaryWriter(log_dir="runs/"+USED_PROBLEM+"/"+datetime.now("%x"))
    #
    # csv_file = "training_log.csv"
    #
    # with open(csv_file, "w", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow([
    #         "episode",
    #         "total_reward",
    #         "avg_advantage",
    #         "avg_return"
    #     ])

    env = EAEnv(USED_ALGORITHM, USED_PROBLEM)
    agent = train(env)
    torch.save(agent.model.state_dict(), "ppo_final_model"+USED_PROBLEM+".pth")
    print("ppo_final_model"+USED_PROBLEM+".pth")
