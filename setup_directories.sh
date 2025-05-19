#!/bin/bash

algos=("trpo" "ppo")
envs=("halfcheetah-vel" "halfcheetah-dir" "2d-navigation" "bandit-k10-n100" "bandit-k50-n100" "bandit-k5-n100")
seeds=("10" "200" "500")

for algo in "${algos[@]}"; do
  for env in "${envs[@]}"; do
    for seed in "${seeds[@]}"; do
      results_dir="results/$algo/$env/$seed"
      weights_dir="results/$algo/$env/$seed"
      mkdir -p "$results_dir"
      mkdir -p "$weights_dir"
      echo "Created: $results_dir and $weights_dir"
    done
  done
done
