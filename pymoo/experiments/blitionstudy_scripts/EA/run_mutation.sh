#!/usr/bin/env bash

BASE="python PPO_con.py"
SEEDS=(33 55 42)

# --- Reward Scale ---
for reward_scale in 0.2 0.5 0.8 1.0 1.5 2.0; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --reward_scale "$reward_scale" \
      --seed "$seed" \
      --experiment_name "reward_${reward_scale}_s${seed}" \
      | tee "runs/logs/reward_${reward_scale}_s${seed}.log"
  done
done