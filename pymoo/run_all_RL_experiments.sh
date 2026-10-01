#!/usr/bin/env bash

echo "All experiments completed."ed."

for script in ./experiments/ablitionstudy_scripts/RL/*.sh; do
    echo "========================================"
    echo "Running RL: $script"
    echo "========================================"

     if bash "$script"; then
        echo "SUCCESS: $script"
    else
        echo "FAILED: $script"
    fi

    echo "Finished: $script"
    echo
done

echo "All experiments completeded."
