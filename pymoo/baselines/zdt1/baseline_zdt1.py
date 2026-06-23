from pymoo.problems import get_problem
from pymoo.visualization.util import plot

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.nsga3 import NSGA3
USED_SEED = 4
USED_PROBLEM_NAME = "ZDT1"
#########|Baseline1

NSGAII = NSGA2
NSGAII.set_params(seed=USED_SEED)
#############baseline2
NSGAIII = NSGA3
NSGAIII.set_params(seed=USED_SEED)



problem = get_problem(USED_PROBLEM_NAME)
plot(problem.pareto_front(), no_fill=True)

