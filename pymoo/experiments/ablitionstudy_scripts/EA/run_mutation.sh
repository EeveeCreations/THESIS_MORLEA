#!/usr/bin/env bash

BASE="python PPO_con.py"
SEEDS=(33 55 42)

# --- Reward Scale ---
for mutation_probability in 0.2 0.5 0.8 1.0 1.5 2.0; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --mutation_probability "$mutation_probability" \
      --seed "$seed" \
      --experiment_name "mutationprob_${mutation_probability}_s${seed}" \
      | tee "runs/logs/mutationprob_${mutation_probability}_s${seed}.log"
  done
done