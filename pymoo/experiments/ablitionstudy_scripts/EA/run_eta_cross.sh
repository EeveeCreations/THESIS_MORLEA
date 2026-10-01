#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- ETA Crossover ---
for eta in 5 10 15 20 30 50; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --eta_crossover "$eta" \
      --seed "$seed" \
      --experiment_name "eta_cross_${eta}_s${seed}" \
      | tee "runs/logs/eta_cross_${eta}_s${seed}.log"
  done
done