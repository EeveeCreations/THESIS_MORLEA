#!/usr/bin/env bash

BASE="python ppo_continuous_action_pattern.py"
SEEDS=(33 55 42)

TASK="pattern_postition_task"

# --- PPO Clip ---
for clip in 0.05 0.1 0.2 0.3; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --clip "$clip" \
      --seed "$seed" \
      --experiment_name "clip_${clip}_s${seed}" \
      | tee "runs/logs/clip_${clip}_s${seed}.log"
  done
done