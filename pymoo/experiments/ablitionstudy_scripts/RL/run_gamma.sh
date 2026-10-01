#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- PPO Gamma ---
for gamma in 0.90 0.95 0.97 0.99 0.995; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --gamma "$gamma" \
      --seed "$seed" \
      --experiment_name "gamma_${gamma}_s${seed}" \
      | tee "runs/logs/gamma_${gamma}_s${seed}.log"
  done
done