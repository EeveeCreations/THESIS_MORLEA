#!/usr/bin/env bash

set -e

for script in  .experiments/blitionstudy_scripts/EA/*.sh; do
    echo "========================================"
    echo "Running EA: $script"
    echo "========================================"

     if bash "$script"; then
        echo "SUCCESS: $script"
    else
        echo "FAILED: $script"
    fi

    echo "Finished: $script"
    echo
done

echo "All experiments completed."