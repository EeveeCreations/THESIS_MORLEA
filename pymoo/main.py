from pymoo.problems import get_problem
from pymoo.visualization.util import plot

problem = get_problem("zdt1")
plot(problem.pareto_front(), no_fill=True)
