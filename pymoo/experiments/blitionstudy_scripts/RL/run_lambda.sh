#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- PPO Lambda ---
for lambda in 0.90 0.95 0.97 0.98 0.99; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --lambda "$lambda" \
      --seed "$seed" \
      --experiment_name "lambda_${lambda}_s${seed}" \
      | tee "runs/logs/lambda_${lambda}_s${seed}.log"
  done
done