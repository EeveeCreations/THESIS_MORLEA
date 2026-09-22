import argparse

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.problems import get_problem

import numpy as np
from torch import optim

from ACTOR_CRIT_CON import ActorCritic


############ Basic  Soltion functiom
USED_SEED = 55 ## 33, 55, 42
### Main   Dynamic Parameters N#############################################################################
USED_PROBLEM_NAME= "zdt6"
USE_ALGORITHM = "MOEA_RL"


### EVO ALGO NSGAII STATICsop[ ##################
CROSSOVER_PROBABILITY = 0.5
MUTATION_PROBABILITY = 0.5


#ETA is used to control the spread or variation strength in genetic operators
ETA_CROSSOVER = 20
ETA_MUTATION = 15

TRUNCATION_CONDITION =  0.001
MIN_IMPROVEMENT = 0.2

MAX_GENERATIONS = 200
POP_SIZE = 1000


#HYPER VOLUME REFRENCE POINT###############################################################
REF_POINT = np.array([1.1, 1.1])


#######ENVIRONMENT PARAMTERS  ###########################################################
REWARD_SCALE = 0.8

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

# ================================================
# Overrriddeessss
# ================================================

parser = argparse.ArgumentParser()

parser.add_argument("--seed", type=int, default=USED_SEED)
parser.add_argument("--crossover_probability", type=float, default=CROSSOVER_PROBABILITY)
parser.add_argument("--mutation_probability", type=float, default=MUTATION_PROBABILITY)
parser.add_argument("--eta_crossover", type=float, default=ETA_CROSSOVER)
parser.add_argument("--eta_mutation", type=float, default=ETA_MUTATION)

parser.add_argument("--reward_scale", type=float, default=REWARD_SCALE)
parser.add_argument("--min_improvement", type=float, default=MIN_IMPROVEMENT)

parser.add_argument("--gamma", type=float, default=GAMMA)
parser.add_argument("--lambda", type=float, default=LAMBDA)
parser.add_argument("--clip", type=float, default=CLIP)
parser.add_argument("--learning_rate", type=float, default=LEARNING_RATE)
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--entropy_count", type=float, default=ENTHROPHY_COUNT)


args, unkown = parser.parse_known_args()

# ============================================================
# and apply if needed ^w^
# ============================================================

if args.seed is not None:
    USED_SEED = args.seed

if args.problem is not None:
    USED_PROBLEM_NAME = args.problem

if args.crossover_probability is not None:
    CROSSOVER_PROBABILITY = args.crossover_probability

if args.mutation_probability is not None:
    MUTATION_PROBABILITY = args.mutation_probability

if args.eta_crossover is not None:
    ETA_CROSSOVER = args.eta_crossover

if args.eta_mutation is not None:
    ETA_MUTATION = args.eta_mutation

if args.truncation_condition is not None:
    TRUNCATION_CONDITION = args.truncation_condition

if args.min_improvement is not None:
    MIN_IMPROVEMENT = args.min_improvement

if args.max_generations is not None:
    MAX_GENERATIONS = args.max_generations

if args.pop_size is not None:
    POP_SIZE = args.pop_size

if args.reward_scale is not None:
    REWARD_SCALE = args.reward_scale

if args.gamma is not None:
    GAMMA = args.gamma

if args.lambda_ is not None:
    LAMBDA = args.lambda_

if args.clip is not None:
    CLIP = args.clip

if args.learning_rate is not None:
    LEARNING_RATE = args.learning_rate

if args.epochs is not None:
    EPOCHS = args.epochs

if args.entropy_count is not None:
    ENTHROPHY_COUNT = args.entropy_count

if args.actor_loss is not None:
    ACTOR_LOSS = args.actor_loss

### PROBLEM  / ALGORITHEM USED
USED_PROBLEM = get_problem(USED_PROBLEM_NAME)
USED_ALGORITHM = NSGA2(
    crossover=SBX(eta=ETA_CROSSOVER, prob=CROSSOVER_PROBABILITY),
    mutation=PM(eta=ETA_MUTATION, prob=MUTATION_PROBABILITY),
    pop_size=POP_SIZE)
USED_ALGORITHM.setup(USED_PROBLEM, seed=USED_SEED)
FINAL_RUNN_NAME = str("ppo_final_model" + USED_PROBLEM_NAME + "_MAX_GENERATIONS" + str(MAX_GENERATIONS) +
                      "_POP_SIZE" + str(POP_SIZE) + "_ETA_CROSSOVER" + str(ETA_CROSSOVER) +
                      "_REWARD_SCALE" + str(REWARD_SCALE) + "_ETA_MUTATION" + str(ETA_MUTATION) +
                      "_MIN_IMPROVEMENT" + str(MIN_IMPROVEMENT) + "_USED_SEED" + str(USED_SEED)
                      )