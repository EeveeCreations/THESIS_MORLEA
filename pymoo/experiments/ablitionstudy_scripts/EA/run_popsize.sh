#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- Crossover Probability ---
for pop_size in 0.3 0.5 0.7 0.9; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --pop_size "$pop_size" \
      --seed "$seed" \
      --experiment_name "popsize_${pop_size}_s${seed}" \
      | tee "runs/logs/popsize_${pop_size}_s${seed}.log"
  done
done