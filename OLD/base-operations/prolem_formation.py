import numpy as np
import torch
import matplotlib.pyplot as plt
import random


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

# Reference point for hypervolume (worse than any expected value)
HV_REFERENCE = np.array([100.0, 100.0])


# ++++++++++++++++++++++++++++++++++++++++++++
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
    plt.savefig("/mnt/user-data/outputs/results.png", dpi=150, bbox_inches='tight')
    print("\n  Plot saved → /mnt/user-data/outputs/results.png")
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
    print("  Plot saved → /outputs/pareto_fronts.png")
    plt.show()


#  ++++++++++++++++++++++++++++++++++++++++++++
#  MAIN
#  ++++++++++++++++++++++++++++++++++++++++++++

if __name__ == "__main__":
    print("=" * 60)
    print(" RL Parameter Tuning — Multi-Objective EA Demo")
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