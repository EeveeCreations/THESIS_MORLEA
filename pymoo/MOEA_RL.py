from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.problems import get_problem

import numpy as np
from torch import optim

from ACTOR_CRIT_CON import ActorCritic


############ Basic  Soltion functiom
USED_SEED = 42
### Main   Dynamic Parameters N#############################################################################
USED_PROBLEM_NAME= "zdt4"

### EVO ALGO NSGAII ##################
CROSSOVER_PROBABILITY = 0.9
MUTATION_PROBABILITY = 0.9

ETA_CROSSOVER = 15
ETA_MUTATION = 20

MAX_GENERATIONS=200
POP_SIZE =1000


#HYPER VOLUME REFRENCE POINT###############################################################
REF_POINT = np.array([1.1, 1.1])


#######ENVIRONMENT PARAMTERS  ###########################################################
REWARD_SCALE = 0.5

#### Parameters RL #######################################################################
RL_GAMMA = 0.99
RL_LR =  1e-3
RL_EPSILON =0.01
##### Q-Table
QT_EPISODES=500

GAMMA = 0.97
LAMBDA= 0.98
CLIP= 0.005
LEARNING_RATE= 2e-4
EPOCHS = 20
ENTHROPHY_COUNT = 0.1
ACTOR_LOSS = 0.8
MODEL = ActorCritic
OPTIMIZER= optim.Adam

### PROBLEM  / ALGORITHEM USED
USED_PROBLEM = get_problem(USED_PROBLEM_NAME)
USED_ALGORITHM = NSGA2(
    crossover=SBX(eta=ETA_CROSSOVER, prob=CROSSOVER_PROBABILITY),
    mutation=PM(eta=ETA_MUTATION, prob=MUTATION_PROBABILITY),
    pop_size=POP_SIZE)
USED_ALGORITHM.setup(USED_PROBLEM, seed=USED_SEED)


###################### SINGLE USE FUNCTIONS NOT  USD  FOR  OPTIMIZTING OR  ACTUAL THESIS  RESERACH
# termination = get_termination("n_gen", max_generations)
# ### Optimize
# results = minimize(
#     USED_PROBLEM,
#     USED_ALGORITHM,
#     termination,
#     seed=1,
#     verbose=True
# )
#
#
# F = results.F
#
# print("Number of Pareto solutions found:", len(F))
#
#


# pf = USED_PROBLEM.pareto_front()
#
# def plot_pareto_front(ea_algo, rl_algo, problem_name, pareto_front):
#     plt.figure()
#     plt.scatter(F[:, 0], F[:, 1], label=str(ea_algo +" Approximation"))
#     plt.plot(pareto_front[:, 0], pareto_front[:, 1], color="red", label="True Pareto Front")
#     plt.xlabel("f1")
#     plt.ylabel("f2")
#     plt.title(str(ea_algo+ " on "+ problem_name))
#     plt.legend()
#     plt.grid(True)
#     plt.show()
#
#
# plot_pareto_front("NSGAII","", USED_PROBLEM_NAME, pf)
