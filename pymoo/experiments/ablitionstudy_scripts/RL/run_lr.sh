
#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- PPO Learning Rate ---
for lr in 1e-5 5e-5 1e-4 2e-4 5e-4 1e-3; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --learning_rate "$lr" \
      --seed "$seed" \
      --experiment_name "lr_${lr}_s${seed}" \
      | tee "runs/logs/lr_${lr}_s${seed}.log"
  done
done