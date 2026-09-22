#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- PPO Entropy ---
for entropy in 0.0 0.01 0.05 0.1 0.2; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --entropy_count "$entropy" \
      --seed "$seed" \
      --experiment_name "entropy_${entropy}_s${seed}" \
      | tee "runs/logs/entropy_${entropy}_s${seed}.log"
  done
done