
#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- PPO Epochs ---
for epochs in 5 10 20 30 40; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --epochs "$epochs" \
      --seed "$seed" \
      --experiment_name "epochs_${epochs}_s${seed}" \
      | tee "runs/logs/epochs_${epochs}_s${seed}.log"
  done
done