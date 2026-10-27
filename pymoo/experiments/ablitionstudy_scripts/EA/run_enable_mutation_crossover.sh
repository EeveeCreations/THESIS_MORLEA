#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- Enable Mutation/ crossover  / early termination ---
for early_termination in True False; do
  for enable_mutation in True False; do
    for enable_crossover in True False; do
      for seed in "${SEEDS[@]}"; do
        $BASE --task "$TASK" \
          --enable_mutation "$enable_mutation" \
          --enable_crossover "$enable_crossover" \
          --early_termination "$early_termination"
          --seed "$seed" \
          --experiment_name "enable_mut_cross_earlyterm_${enable_mutation}_${enable_crossover}_${early_termination}_s${seed}" \
          | tee "runs/logs/enable_mut_cross_earlyterm__${enable_mutation}_${enable_crossover}_${early_termination}_s${seed}.log"
      done
    done
  done
done