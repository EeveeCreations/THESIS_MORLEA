import argparse

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

args = parser.parse_args()