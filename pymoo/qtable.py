from datetime import datetime
from MOEA_RL import USED_PROBLEM, USED_ALGORITHEM
import numpy as np
import random
import sys

from ea_environment import EAEnv

def q_learning(env, episodes=500, alpha=0.1, gamma=0.99, epsilon=0.1):

    n_actions = 4
    Q = {}

    def get_q(state):
        state = tuple(state)
        print(type(state),  state)
        print(type(Q),  Q)

        if state not in Q:
            Q[state] = np.zeros(n_actions)
        return Q[state]

    for episode in range(episodes):

        state = tuple(env.reset())
        done = False

        while not done:

            if random.uniform(0,1) < epsilon:
                action = random.randint(0, n_actions-1)
            else:
                action = np.argmax(get_q(state))

            next_state, reward, done = env.step(action)
            next_state = tuple(next_state)

            best_next = np.max(get_q(next_state))

            Q[state][action] += alpha * (
                reward + gamma * best_next - Q[state][action]
            )

            state = next_state

    return Q


### TO CHEC IF IT WORKS TO BE REMOVED

env = EAEnv(USED_ALGORITHEM, USED_PROBLEM)
Q = q_learning(env)
sys.stdout = open("output-PArams"+ datetime.today()+".txt", "w")
print("Learned Q-table:")
for state in Q:
    print(state, Q[state])
sys.stdout.close()