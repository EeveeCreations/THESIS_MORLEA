#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- Enable Mutation/ crossover  / early termination ---  We do 1 /0 mostly cuas ei wrote down intergers for the    RMOA env to recieve as booleans can sometiem sretae  problems
for early_termination in 1 0; do
  for enable_mutation in 1 0; do
    for enable_crossover in 1 0; do
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