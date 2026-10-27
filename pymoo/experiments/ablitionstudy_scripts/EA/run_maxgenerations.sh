#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- Crossover Probability ---
for max_generations in 0.3 0.5 0.7 0.9; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --max_generations "$max_generations" \
      --seed "$seed" \
      --experiment_name "maxgen_${max_generations}_s${seed}" \
      | tee "runs/logs/maxgen_${max_generations}_s${seed}.log"
  done
done