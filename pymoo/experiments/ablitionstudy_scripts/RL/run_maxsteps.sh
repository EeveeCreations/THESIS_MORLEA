
#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)


# --- Reward Scale ---
for max_steps in 0.2 0.5 0.8 1.0 1.5 2.0; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --max_steps "$max_steps" \
      --seed "$seed" \
      --experiment_name "maxsteps_${max_steps}_s${seed}" \
      | tee "runs/logs/maxsteps_${max_steps}_s${seed}.log"
  done
done