import  pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.problems import get_problem
from pymoo.optimize import minimize
from pymoo.termination import get_termination

############ Basic  Soltion functio


### Main   Dynamic Parameters N#############################################################################
USED_PROBLEM="zdt1"

### EVO ALGO NSGAII ##################
crossover_probability = 0.9
mutation_probability = 0.9
max_generations =200

#### Parameters RL ####################
rl_gamma = 0.99
rl_lr =  0.001

##### Q-Table

#####  PPO




problem = get_problem(USED_PROBLEM)
algorithm = NSGA2(
    crossover=SBX(eta=15, prob=crossover_probability),
    mutation=PM(eta=20, prob=mutation_probability),
    pop_size=1000
)

termination = get_termination("n_gen", max_generations)


### Optimize
results = minimize(
    problem,
    algorithm,
    termination,
    seed=1,
    verbose=True
)


F = results.F

print("Number of Pareto solutions found:", len(F))




pf = problem.pareto_front()

def plot_pareto_front(ea_algo, rl_algo, problem_name, pareto_front):
    plt.figure()
    plt.scatter(F[:, 0], F[:, 1], label=str(ea_algo +" Approximation"))
    plt.plot(pareto_front[:, 0], pareto_front[:, 1], color="red", label="True Pareto Front")
    plt.xlabel("f1")
    plt.ylabel("f2")
    plt.title(str(ea_algo+ " on "+ problem_name))
    plt.legend()
    plt.grid(True)
    plt.show()


plot_pareto_front("NSGAII","", USED_PROBLEM, pf)
