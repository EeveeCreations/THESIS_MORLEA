from multiprocessing.spawn import set_executable

import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination


####### Import the diffrent partiesss
# from qtable import q_learning cn

############ Basic  Soltion functiom

USED_SEED = 42
### Main   Dynamic Parameters N#############################################################################
USED_PROBLEM_NAME= "zdt1"

### EVO ALGO NSGAII ##################
crossover_probability = 0.9
mutation_probability = 0.9
max_generations =200

#### Parameters RL ####################
rl_gamma = 0.99
rl_lr =  1e-3
rl_epsilon =0.01
##### Q-Table
qt_episodes=500

#####  PPO
#
GAMMA = 0.99
LAMBDA= 0.95
CLIP= 2e-2
LEARNING_RATE= 2e-4
EPOCHS = 10


### PROBLEM  / ALGORITHEM USED
USED_PROBLEM = get_problem(USED_PROBLEM_NAME)
USED_ALGORITHM = NSGA2(
    crossover=SBX(eta=15, prob=crossover_probability),
    mutation=PM(eta=20, prob=mutation_probability),
    pop_size=1000
)
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
