from itertools import count

import torch
import torch.nn as nn
import math
import random
from collections import deque, namedtuple
import matplotlib.pyplot as plt
from torch import optim

from ea_environment import EAEnv, ACTION_SPACE
from MOEA_RL import USED_PROBLEM, USED_ALGORITHM

### NN Model
FILTERS_IN_NN = 128
FILTERS_OUT_NN = 128
FILTERS_MID_NN = 128

####   DQN  SPECIFIED PARAMETRES
BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 2500
TAU = 0.005
LR = 3e-4

transition = namedtuple('Transition',
                                ('state', 'action', 'next_state', 'reward'))


class ReplayBuffer:
    def __init__(self, capacity=5000):
        self.buffer = deque([],maxlen=capacity)

    def push(self, *args): ## SAFE TRANS
        self.buffer.append(transition(*args))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        # s, a, r, ns, d = zip(*batch)
        # return (torch.FloatTensor(np.array(s)),
        #         torch.LongTensor(a),
        #         torch.FloatTensor(r),
        #         torch.FloatTensor(np.array(ns)),
        #         torch.FloatTensor(d))
        return batch

    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    """Small MLP: state → Q-values for each action."""
    def __init__(self, state_dimensions, n_actions):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dimensions, FILTERS_OUT_NN),
            nn.ReLU(),
            nn.Linear(FILTERS_IN_NN, FILTERS_OUT_NN),
            nn.ReLU(),
            nn.Linear(FILTERS_IN_NN, n_actions),
        )

    def forward(self, x):
        return self.net(x)


env = EAEnv(USED_PROBLEM, USED_ALGORITHM)

n_actions = ACTION_SPACE.__len__()
# Get the number of state observations
state, info = env.reset() #
n_observations = len(state)

policy_net = DQNNetwork(n_observations, n_actions)
#Create a target network for  a  Doubel DQN effect
target_net = DQNNetwork(n_observations, n_actions)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
buffer = ReplayBuffer(10000)


steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[env.action_space.sample()]])


episode_durations = []


def optimize_model():
    if len(buffer) < BATCH_SIZE:
        return
    transitions = buffer.sample(BATCH_SIZE)
    batch = transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)) , dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    next_state_values = torch.zeros(BATCH_SIZE)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


############ PLOT@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@



def plot_durations(show_result=False):
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # Take 100 episode averages and plot them too
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            display.display(plt.gcf())

# RUN  DQN #######################################

if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 600
else:
    num_episodes = 50

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, ).unsqueeze(0)
    for t in count():
        action = select_action(state)
        print(action)
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], )
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, ).unsqueeze(0)

        # Store the transition in memory
        buffer.push(state, action, next_state, reward)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key] * TAU + target_net_state_dict[key] * (1 - TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break

print('Complete')
plot_durations(show_result=True)
plt.ioff()
plt.show()