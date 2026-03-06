"""
=============================================================================
  RL-Driven Dynamic Parameter Tuning for Multi-Objective Evolutionary Algorithms
=============================================================================
  Thesis demo: Comparing Q-Table vs DQN for adaptive EA parameter control.

  PROBLEM:   Multi-objective optimization
             - Objective 1: Rastrigin(x)  → minimize  (multimodal, hard)
             - Objective 2: Sphere(x)     → minimize  (smooth, easy)
             These two objectives conflict slightly — the agent must balance them.

  EA:        Simple NSGA-II-inspired genetic algorithm
             Parameters tuned dynamically: mutation_rate, crossover_rate

  RL STATE:  [gen_progress, hypervolume_delta, diversity, pareto_size, stagnation]
  RL ACTION: Discrete combo of (mutation_rate, crossover_rate)
  RL REWARD: Change in Hypervolume indicator (captures both objectives at once)

  AGENTS:    1) Q-Table  — tabular, discretized state
             2) DQN      — neural network, continuous state input

  USAGE:
      pip install torch numpy matplotlib
      python rl_ea_parameter_tuning.py
=============================================================================
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import deque
import random
import copy

#  ++++++++++++++++++++++++++++++++++++++++++++
#  GLOBAL CONFIG
#  ++++++++++++++++++++++++++++++++++++++++++++
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
random.seed(SEED)

# EA settings
N_DIMS        = 10        # dimensionality of the search space
POP_SIZE      = 60        # population size
N_GENERATIONS = 80        # generations per RL episode
N_EPISODES    = 30        # how many times we re-run the EA (RL training episodes)
X_BOUNDS      = (-5.12, 5.12)

# RL settings
STATE_DIM     = 5         # size of the RL state vector
N_ACTIONS     = 9         # 3 mutation rates × 3 crossover rates (see ACTION_SPACE)

# Discrete action space: all combinations of (mutation_rate, crossover_rate)
MUT_RATES  = [0.01, 0.1, 0.3]
CX_RATES   = [0.5,  0.7, 0.9]
ACTION_SPACE = [(m, c) for m in MUT_RATES for c in CX_RATES]  # 9 actions

# Reference point for hyper  volume (worse than any expected value)
HV_REFERENCE = np.array([100.0, 100.0])


#  ++++++++++++++++++++++++++++++++++++++++++++
#  OBJECTIVE FUNCTIONS
#  ++++++++++++++++++++++++++++++++++++++++++++

def rastrigin(x: np.ndarray) -> float:
    """Multimodal function — many local optima. Global min = 0 at origin."""
    A = 10
    return A * len(x) + np.sum(x**2 - A * np.cos(2 * np.pi * x))


def sphere(x: np.ndarray) -> float:
    """Simple bowl function. Global min = 0 at origin."""
    return np.sum(x**2)


def evaluate(x: np.ndarray) -> np.ndarray:
    """Returns [rastrigin, sphere] — both to be minimized."""
    return np.array([rastrigin(x), sphere(x)])


#  ++++++++++++++++++++++++++++++++++++++++++++
#  MULTI-OBJECTIVE  WITH NSGA II
#  ++++++++++++++++++++++++++++++++++++++++++++

def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """True if solution a dominates b (a is at least as good in all objectives, better in one)."""
    return np.all(a <= b) and np.any(a < b)


def fast_nondominated_sort(fitnesses: np.ndarray) -> list:
    """Returns list of Pareto fronts (indices). Front 0 = best."""
    n = len(fitnesses)
    domination_count = np.zeros(n, dtype=int)
    dominated_by = [[] for _ in range(n)]
    fronts = [[]]

    for i in range(n):
        for j in range(i + 1, n):
            if dominates(fitnesses[i], fitnesses[j]):
                dominated_by[i].append(j)
                domination_count[j] += 1
            elif dominates(fitnesses[j], fitnesses[i]):
                dominated_by[j].append(i)
                domination_count[i] += 1
        if domination_count[i] == 0:
            fronts[0].append(i)

    current_front = 0
    # print(fronts)
    while len(fronts) <= current_front-1 or fronts[current_front]:
        next_front = []
        for i in fronts[current_front]:
            # print(fronts[current_front], i)
            for j in dominated_by[i]:
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        current_front += 1
        if next_front:
            fronts.append(next_front)
        if len(fronts) > current_front:
            break

    return fronts


def crowding_distance(fitnesses: np.ndarray, front: list) -> np.ndarray:
    """Crowding distance for diversity preservation within a front."""
    n = len(front)
    distances = np.zeros(n)
    if n <= 2:
        distances[:] = np.inf
        return distances

    for obj in range(fitnesses.shape[1]):
        sorted_idx = np.argsort(fitnesses[front, obj])
        obj_range = fitnesses[front[sorted_idx[-1]], obj] - fitnesses[front[sorted_idx[0]], obj]
        if obj_range == 0:
            continue
        distances[sorted_idx[0]] = np.inf
        distances[sorted_idx[-1]] = np.inf
        for k in range(1, n - 1):
            distances[sorted_idx[k]] += (
                fitnesses[front[sorted_idx[k + 1]], obj] -
                fitnesses[front[sorted_idx[k - 1]], obj]
            ) / obj_range

    return distances


def select_parents(population, fitnesses, n_select):
    """Tournament selection based on Pareto rank and crowding distance."""
    fronts = fast_nondominated_sort(fitnesses)
    rank = np.zeros(len(population), dtype=int)
    for r, front in enumerate(fronts):
        for idx in front:
            rank[idx] = r

    selected = []
    for _ in range(n_select):
        i, j = np.random.randint(0, len(population), 2)
        if rank[i] < rank[j]:
            selected.append(i)
        elif rank[j] < rank[i]:
            selected.append(j)
        else:
            selected.append(i if np.random.rand() < 0.5 else j)
    return selected


def crossover(p1, p2, cx_rate):
    """Simulated Binary Crossover (SBX)."""
    if np.random.rand() > cx_rate:
        return p1.copy(), p2.copy()
    eta = 15
    u = np.random.rand(len(p1))
    beta = np.where(u <= 0.5,
                    (2 * u) ** (1 / (eta + 1)),
                    (1 / (2 * (1 - u))) ** (1 / (eta + 1)))
    c1 = 0.5 * ((1 + beta) * p1 + (1 - beta) * p2)
    c2 = 0.5 * ((1 - beta) * p1 + (1 + beta) * p2)
    return np.clip(c1, *X_BOUNDS), np.clip(c2, *X_BOUNDS)


def mutate(x, mut_rate):
    """Polynomial mutation."""
    eta = 20
    mutant = x.copy()
    for i in range(len(x)):
        if np.random.rand() < mut_rate:
            u = np.random.rand()
            low, high = X_BOUNDS
            delta = high - low
            if u < 0.5:
                delta_q = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta_q = 1 - (2 * (1 - u)) ** (1 / (eta + 1))
            mutant[i] = np.clip(x[i] + delta_q * delta, low, high)
    return mutant


def hypervolume_2d(pareto_fitnesses: np.ndarray, ref: np.ndarray) -> float:
    """Exact 2D hypervolume calculation."""
    points = pareto_fitnesses[np.all(pareto_fitnesses < ref, axis=1)]
    if len(points) == 0:
        return 0.0
    sorted_points = points[np.argsort(points[:, 0])]
    hv = 0.0
    prev_y = ref[1]
    for p in sorted_points:
        hv += (ref[0] - p[0]) * (prev_y - p[1])
        prev_y = p[1]
    return max(hv, 0.0)


class MultiObjectiveEA:
    """
    NSGA-II style EA with externally controlled mutation and crossover rates.
    The RL agent calls .step() each generation, passing in the current parameters.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.population = np.random.uniform(*X_BOUNDS, (POP_SIZE, N_DIMS))
        self.fitnesses   = np.array([evaluate(x) for x in self.population])
        self.generation  = 0
        self.stagnation  = 0
        self.prev_hv     = 0.0
        self.hv_history  = []
        self._update_pareto()
        return self._get_state()

    def _update_pareto(self):
        fronts = fast_nondominated_sort(self.fitnesses)
        self.pareto_front_idx = fronts[0]
        self.pareto_fitnesses  = self.fitnesses[self.pareto_front_idx]
        self.hv = hypervolume_2d(self.pareto_fitnesses, HV_REFERENCE)
        self.hv_history.append(self.hv)

    def _get_state(self) -> np.ndarray:
        """
        5-dimensional state vector for the RL agent:
          [0] generation progress  (0→1)
          [1] hypervolume delta    (normalised improvement)
          [2] population diversity (mean pairwise distance, normalised)
          [3] pareto front size    (fraction of population)
          [4] stagnation count     (normalised)
        """
        gen_progress  = self.generation / N_GENERATIONS
        hv_delta      = np.clip((self.hv - self.prev_hv) / (self.prev_hv + 1e-9), -1, 1)
        diversity     = np.mean(np.std(self.population, axis=0)) / (X_BOUNDS[1] - X_BOUNDS[0])
        pareto_size   = len(self.pareto_front_idx) / POP_SIZE
        stagnation    = np.clip(self.stagnation / 10.0, 0, 1)
        return np.array([gen_progress, hv_delta, diversity, pareto_size, stagnation],
                        dtype=np.float32)

    def step(self, mut_rate: float, cx_rate: float):
        """Run one generation with given parameters. Returns (next_state, reward, done)."""
        self.prev_hv = self.hv

        # --- Generate offspring ---
        parent_idx = select_parents(self.population, self.fitnesses, POP_SIZE)
        offspring  = []
        for k in range(0, POP_SIZE, 2):
            p1 = self.population[parent_idx[k]]
            p2 = self.population[parent_idx[min(k + 1, POP_SIZE - 1)]]
            c1, c2 = crossover(p1, p2, cx_rate)
            offspring.extend([mutate(c1, mut_rate), mutate(c2, mut_rate)])
        offspring = np.array(offspring[:POP_SIZE])
        offspring_fit = np.array([evaluate(x) for x in offspring])

        # --- NSGA-II selection: keep best POP_SIZE from parents + offspring ---
        combined_pop = np.vstack([self.population, offspring])
        combined_fit = np.vstack([self.fitnesses,  offspring_fit])
        fronts = fast_nondominated_sort(combined_fit)

        new_pop_idx = []
        for front in fronts:
            if len(new_pop_idx) + len(front) <= POP_SIZE:
                new_pop_idx.extend(front)
            else:
                remaining = POP_SIZE - len(new_pop_idx)
                cd = crowding_distance(combined_fit, front)
                sorted_by_cd = [front[i] for i in np.argsort(-cd)[:remaining]]
                new_pop_idx.extend(sorted_by_cd)
                break

        self.population = combined_pop[new_pop_idx]
        self.fitnesses   = combined_fit[new_pop_idx]
        self.generation += 1
        self._update_pareto()

        # --- Stagnation tracking ---
        if abs(self.hv - self.prev_hv) < 1e-6:
            self.stagnation += 1
        else:
            self.stagnation = 0

        # --- Reward = hypervolume improvement ---
        reward = (self.hv - self.prev_hv) / (self.prev_hv + 1e-9)
        reward = np.clip(reward, -1.0, 1.0)

        done  = (self.generation >= N_GENERATIONS)
        state = self._get_state()
        return state, float(reward), done


#  ++++++++++++++++++++++++++++++++++++++++++++
#  AGENT 1: Q-TABLE
#  ++++++++++++++++++++++++++++++++++++++++++++

class QTableAgent:
    """
    Tabular Q-learning with discretised state.
    State is binned into a fixed grid — trades precision for simplicity.
    """

    def __init__(self, n_bins=4, lr=0.1, gamma=0.95, epsilon=1.0, epsilon_decay=0.97):
        self.n_bins       = n_bins
        self.lr           = lr
        self.gamma        = gamma
        self.epsilon      = epsilon
        self.epsilon_min  = 0.05
        self.epsilon_decay= epsilon_decay

        # Bin edges for each state dimension
        self.bins = [
            np.linspace(0, 1, n_bins + 1)[1:-1],    # gen_progress
            np.linspace(-1, 1, n_bins + 1)[1:-1],   # hv_delta
            np.linspace(0, 1, n_bins + 1)[1:-1],    # diversity
            np.linspace(0, 1, n_bins + 1)[1:-1],    # pareto_size
            np.linspace(0, 1, n_bins + 1)[1:-1],    # stagnation
        ]
        # Q-table shape: (n_bins, n_bins, n_bins, n_bins, n_bins, N_ACTIONS)
        self.q_table = np.zeros([n_bins] * STATE_DIM + [N_ACTIONS])
        self.episode_rewards = []

    def _discretize(self, state: np.ndarray) -> tuple:
        return tuple(np.digitize(state[i], self.bins[i]) for i in range(STATE_DIM))

    def select_action(self, state: np.ndarray) -> int:
        if np.random.rand() < self.epsilon:
            return np.random.randint(N_ACTIONS)
        s = self._discretize(state)
        return int(np.argmax(self.q_table[s]))

    def update(self, state, action, reward, next_state, done):
        s  = self._discretize(state)
        ns = self._discretize(next_state)
        td_target = reward + (0 if done else self.gamma * np.max(self.q_table[ns]))
        self.q_table[s][action] += self.lr * (td_target - self.q_table[s][action])

    def end_episode(self, total_reward):
        self.episode_rewards.append(total_reward)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


#  ++++++++++++++++++++++++++++++++++++++++++++
#  AGENT 2: DQN (Deep Q-Network)
#  ++++++++++++++++++++++++++++++++++++++++++++

class DQNNetwork(nn.Module):
    """Small MLP: state → Q-values for each action."""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(STATE_DIM, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, N_ACTIONS),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    """Experience replay: stores (s, a, r, s', done) tuples."""
    def __init__(self, capacity=5000):
        self.buffer = deque(maxlen=capacity)

    def push(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (torch.FloatTensor(np.array(s)),
                torch.LongTensor(a),
                torch.FloatTensor(r),
                torch.FloatTensor(np.array(ns)),
                torch.FloatTensor(d))

    def __len__(self):
        return len(self.buffer)
#  ++++++++++++++++++++++++++++++++++++++++++++
#  BASELINE: Random parameter selection
#  ++++++++++++++++++++++++++++++++++++++++++++

class RandomAgent:
    """Randomly picks parameters each generation — our baseline."""
    def __init__(self):
        self.episode_rewards = []

    def select_action(self, state):
        return np.random.randint(N_ACTIONS)

    def update(self, *args): pass

    def end_episode(self, total_reward):
        self.episode_rewards.append(total_reward)


#  ++++++++++++++++++++++++++++++++++++++++++++
#  TRAINING LOOP
#  ++++++++++++++++++++++++++++++++++++++++++++

def run_experiment(agent, label: str, verbose=True):
    """
    Train an agent for N_EPISODES.
    Returns: episode rewards, final HV histories per episode.
    """
    ea = MultiObjectiveEA()
    all_hv_histories = []

    for ep in range(N_EPISODES):
        state = ea.reset()
        total_reward = 0.0

        for gen in range(N_GENERATIONS):
            action = agent.select_action(state)
            mut_rate, cx_rate = ACTION_SPACE[action]
            next_state, reward, done = ea.step(mut_rate, cx_rate)
            agent.update(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                break

        all_hv_histories.append(ea.hv_history.copy())
        agent.end_episode(total_reward)

        if verbose and (ep + 1) % 5 == 0:
            avg_hv = np.mean([h[-1] for h in all_hv_histories[-5:]])
            eps    = getattr(agent, 'epsilon', '-')
            print(f"  [{label}] Episode {ep+1:3d}/{N_EPISODES} | "
                  f"Reward: {total_reward:7.3f} | Final HV: {ea.hv:.2f} | "
                  f"ε: {eps:.3f}" if isinstance(eps, float) else
                  f"  [{label}] Episode {ep+1:3d}/{N_EPISODES} | "
                  f"Reward: {total_reward:7.3f} | Final HV: {ea.hv:.2f}")

    return all_hv_histories


#  ++++++++++++++++++++++++++++++++++++++++++++
#  PLOTTING
#  ++++++++++++++++++++++++++++++++++++++++++++

def smooth(values, window=5):
    return np.convolve(values, np.ones(window) / window, mode='valid')


def plot_results(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("RL-Driven Parameter Tuning for Multi-Objective EA\n"
                 "(Rastrigin + Sphere, 10D)", fontsize=13, fontweight='bold')

    colors = {'Q-Table': '#2196F3', 'DQN': '#E91E63', 'Random': '#9E9E9E'}

    # ── Plot 1: Episode rewards over training ──
    ax = axes[0]
    for label, (hv_hist, agent) in results.items():
        rewards = agent.episode_rewards
        if len(rewards) > 5:
            ax.plot(smooth(rewards), label=label, color=colors[label], linewidth=2)
        else:
            ax.plot(rewards, label=label, color=colors[label], linewidth=2)
    ax.set_title("Learning Curve (Episode Reward)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Reward")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 2: Final hypervolume per episode ──
    ax = axes[1]
    for label, (hv_hist, agent) in results.items():
        final_hvs = [h[-1] for h in hv_hist]
        if len(final_hvs) > 5:
            ax.plot(smooth(final_hvs), label=label, color=colors[label], linewidth=2)
        else:
            ax.plot(final_hvs, label=label, color=colors[label], linewidth=2)
    ax.set_title("Final Hypervolume per Episode\n(higher = better Pareto front)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Hypervolume")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # ── Plot 3: HV growth during last episode (all agents) ──
    ax = axes[2]
    for label, (hv_hist, agent) in results.items():
        last_hv = hv_hist[-1]
        ax.plot(last_hv, label=label, color=colors[label], linewidth=2)
    ax.set_title("HV Growth During Final Episode\n(convergence behaviour)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Hypervolume")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/outputs/results.png", dpi=150, bbox_inches='tight')
    print("\n  Plot saved → /outputs/results.png")
    plt.show()


def plot_pareto_fronts(ea_qtable, ea_dqn, ea_random):
    """Show the Pareto fronts achieved by each agent's final episode."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Pareto Fronts — Final Episode\n(lower-left = better)", fontsize=12)

    for ax, ea, label, color in zip(
        axes,
        [ea_qtable, ea_dqn, ea_random],
        ['Q-Table', 'DQN', 'Random'],
        ['#2196F3', '#E91E63', '#9E9E9E']
    ):
        pf = ea.pareto_fitnesses
        ax.scatter(ea.fitnesses[:, 0], ea.fitnesses[:, 1],
                   alpha=0.3, color=color, s=20, label='Population')
        ax.scatter(pf[:, 0], pf[:, 1],
                   color=color, s=60, edgecolors='black', zorder=5, label='Pareto front')
        ax.set_title(f"{label}\nPareto size: {len(pf)}  HV: {ea.hv:.1f}")
        ax.set_xlabel("Rastrigin (obj 1)")
        ax.set_ylabel("Sphere (obj 2)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("/outputs/pareto_fronts.png", dpi=150, bbox_inches='tight')
    print(" Plot saved → /outputs/pareto_fronts.png")
    plt.show()


#  ++++++++++++++++++++++++++++++++++++++++++++
#  MAIN
#  ++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    print("=" * 60)
    print(" RL Parameter Tuning — Multi-Objective EA")
    print("=" * 60)
    print(f" Dimensions: {N_DIMS}  |  Pop size: {POP_SIZE}")
    print(f" Generations/episode: {N_GENERATIONS}  |  Episodes: {N_EPISODES}")
    print(f" Action space: {N_ACTIONS} combos of (mut_rate × cx_rate)")
    print("=" * 60)

    # ── Run all three agents ──
    print("\n  Training Q-Table agent...")
    qtable_agent = QTableAgent()
    qtable_hv    = run_experiment(qtable_agent, "Q-Table")

    print("\n  Training DQN agent...")
    dqn_agent = DQNAgent()
    dqn_hv    = run_experiment(dqn_agent, "DQN")

    print("\n  Running Random baseline...")
    random_agent = RandomAgent()
    random_hv    = run_experiment(random_agent, "Random", verbose=False)

    # ── Summary ──
    print("\n" + "=" * 60)
    print(" RESULTS SUMMARY (avg final HV over last 5 episodes)")
    print("=" * 60)
    for label, hv_hist in [("Q-Table", qtable_hv), ("DQN", dqn_hv), ("Random", random_hv)]:
        avg = np.mean([h[-1] for h in hv_hist[-5:]])
        print(f"  {label:10s}:  {avg:.2f}")

    # ── Plot learning curves ──
    results = {
        'Q-Table': (qtable_hv, qtable_agent),
        'DQN':     (dqn_hv,    dqn_agent),
        'Random':  (random_hv, random_agent),
    }
    plot_results(results)

    # ── Run one final episode for each agent and plot Pareto fronts ──
    print("\n  Running final episode for Pareto front comparison...")

    def final_run(agent):
        ea = MultiObjectiveEA()
        state = ea.reset()
        for _ in range(N_GENERATIONS):
            action = agent.select_action(state)
            mut_rate, cx_rate = ACTION_SPACE[action]
            state, _, done = ea.step(mut_rate, cx_rate)
            if done:
                break
        return ea

    # Set epsilon to 0 (exploit, don't explore) for final run
    qtable_agent.epsilon = 0.0
    dqn_agent.epsilon    = 0.0

    ea_q = final_run(qtable_agent)
    ea_d = final_run(dqn_agent)
    ea_r = final_run(random_agent)

    plot_pareto_fronts(ea_q, ea_d, ea_r)

    print("\n✓ Done. Check outputs:")
    print("   results.png       — learning curves & HV comparison")
    print("   pareto_fronts.png — Pareto front quality comparison")