
#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)


# --- Reward Scale ---
for min_steps in 0.2 0.5 0.8 1.0 1.5 2.0; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --min_steps "$min_steps" \
      --seed "$seed" \
      --experiment_name "minsteps_${min_steps}_s${seed}" \
      | tee "runs/logs/minsteps_${min_steps}_s${seed}.log"
  done
done