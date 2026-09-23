#!/usr/bin/env bash

set -e

    echo "========================================"
    echo "Running EVERYTHING: $script"
    echo "========================================"

for script in .experiments/blitionstudy_scripts/*.sh; do
    if bash "$script"; then
        echo "SUCCESS: $script"
    else
        echo "FAILED: $script"
    fi
    echo
done

echo "All experiments complet"


#set -e
#
#for script in EA/*.sh; do
#    echo "========================================"
#    echo "Running EA: $script"
#    echo "========================================"
#
#     if bash "$script"; then
#        echo "SUCCESS: $script"
#    else
#        echo "FAILED: $script"
#    fi
#
#    echo "Finished: $script"
#    echo
#done
#
#echo "All experiments completed."ed."
#
#for script in EA/*.sh; do
#    echo "========================================"
#    echo "Running RL: $script"
#    echo "========================================"
#
#     if bash "$script"; then
#        echo "SUCCESS: $script"
#    else
#        echo "FAILED: $script"
#    fi
#
#    echo "Finished: $script"
#    echo
#done
#
#echo "All experiments completed."ed."