from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.problems import get_problem
from pymoo.visualization.util import plot

problem = get_problem("zdt1")

BASE_ALGORITHEM =  NSGA2(
    crossover=SBX(eta=15, prob=crossover_probability),
    mutation=PM(eta=20, prob=mutation_probability),
    pop_size=1000
)

plot(problem.pareto_front(), no_fill=True)
