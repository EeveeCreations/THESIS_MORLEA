#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- Crossover Probability ---
for crossover_prob in 0.3 0.5 0.7 0.9; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --crossover_probability "$crossover_prob" \
      --seed "$seed" \
      --experiment_name "crossprob_${crossover_prob}_s${seed}" \
      | tee "runs/logs/crossprob_${crossover_prob}_s${seed}.log"
  done
done