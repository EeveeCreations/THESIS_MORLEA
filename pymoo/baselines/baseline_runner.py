import os
import numpy as np
from matplotlib import pyplot as plt
from pymoo.optimize import minimize
from pymoo.problems import get_problem
from pymoo.visualization.util import plot

from pymoo.algorithms.moo.nsga2 import NSGA2
from torch.utils.tensorboard import SummaryWriter

SEEDS = [33, 55,  42]
PROBLEMS = ["ZDT1","ZDT4","ZDT6"]
#########|Baseline1

def save_events(res, problem_name, seed):
    log_dir = f"logs/{problem_name}/seed_{seed}"
    writer = SummaryWriter(log_dir)

    for i, generation in enumerate(res.history):
        F = generation.pop.get("F")

        writer.add_scalar(
            "best_objective_1",
            np.min(F[:, 0]),
            i
        )

        writer.add_scalar(
            "best_objective_2",
            np.min(F[:, 1]),
            i
        )

    writer.close()

def run_baseline(problem_name, seed):
    NSGAII = NSGA2()
    problem = get_problem(problem_name)
    plot(problem.pareto_front(), no_fill=True)
    NSGAII.setup(problem,seed=seed)


    res = minimize(
        problem,
        NSGAII,
        termination=("n_gen", 200),
        seed=seed,
        save_history=True,
        verbose=False
    )

    save_events(res, problem_name, seed)
    # True Pareto front
    pf = problem.pareto_front()

    # Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(
        res.F[:, 0], res.F[:, 1],
        label="NSGA-II solutions"
    )
    plt.plot(
        pf[:, 0], pf[:, 1],
        "r-", label="True Pareto front"
    )

    plt.xlabel("Objective 1")
    plt.ylabel("Objective 2")
    plt.title(f"{problem_name} - Seed {seed}")
    plt.legend()
    plt.grid(True)

    # Save figure
    os.makedirs("results", exist_ok=True)
    plt.savefig(
        f"results/{problem_name}_seed_{seed}.png",
        dpi=300,
        bbox_inches="tight"
    )



if __name__ == '__main__':
    for seed in SEEDS:
        run_baseline(PROBLEMS[0], seed)
        run_baseline(PROBLEMS[1], seed)
        run_baseline(PROBLEMS[2], seed)