
#!/usr/bin/env bash

BASE="python PPO_CON.py"
SEEDS=(33 55 42)

# --- Minimum Improvement ---
for min_improvement in 0.05 0.1 0.2 0.3 0.5; do
  for seed in "${SEEDS[@]}"; do
    $BASE --task "$TASK" \
      --min_improvement "$min_improvement" \
      --seed "$seed" \
      --experiment_name "minimp_${min_improvement}_s${seed}" \
      | tee "runs/logs/minimp_${min_improvement}_s${seed}.log"
  done
done
