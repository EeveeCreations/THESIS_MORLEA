import  numpy as np

from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.algorithms.moo.nsga2 import NSGA2


from pymoo.indicators.hv import HV
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.util.ref_dirs import get_reference_directions
from torch.utils.tensorboard import SummaryWriter

from MOEA_RL import (USED_PROBLEM_NAME, POP_SIZE,
                     CROSSOVER_PROBABILITY, MUTATION_PROBABILITY,
                     ETA_MUTATION,ETA_CROSSOVER,
                     SBX, PM, USED_SEED,MAX_GENERATIONS,
                     REF_POINT)

problem = get_problem(USED_PROBLEM_NAME)


hv = HV(ref_point=REF_POINT)


#########|Baseline1
ref_dirs = get_reference_directions(
    "das-dennis",
    problem.n_obj,
    n_partitions=99
)
NSGAIII = NSGA3( crossover=SBX(eta=ETA_CROSSOVER, prob=CROSSOVER_PROBABILITY),
    mutation=PM(eta=ETA_MUTATION, prob=MUTATION_PROBABILITY),
    pop_size=POP_SIZE, ref_dirs=ref_dirs)

#########|Baseline  2
NSGAII =  NSGA2(
    crossover=SBX(eta=ETA_CROSSOVER, prob=CROSSOVER_PROBABILITY),
    mutation=PM(eta=ETA_MUTATION, prob=MUTATION_PROBABILITY),
    pop_size=POP_SIZE)

algorithms = {
    "NSGAII": NSGAII,
    "NSGAIII": NSGAIII
}

for name, algorithm in algorithms.items():

    writer = SummaryWriter(
        f"runs/BASELINE/{USED_PROBLEM_NAME}/{name}"
    )

    res = minimize(
        problem,
        algorithm,
        ('n_gen', MAX_GENERATIONS),
        seed=USED_SEED,
        save_history=True
    )

    for gen, hist in enumerate(res.history):

        F = hist.pop.get("F")

        writer.add_scalar(
            "Performance/Hypervolume",
            hv(F),
            gen
        )

    writer.close()