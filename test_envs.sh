#!/bin/bash

# List of models and batch sizes
models=("ppo")
batch_sizes=(10 200)

# List of environments and their config folder structure
envs=( "2d-navigation")

for model in "${models[@]}"; do
    cfg_path="weights/${model}"
    out_path="results/${model}"

    for env in "${envs[@]}"; do
        for batch in "${batch_sizes[@]}"; do
            config_path="${cfg_path}/${env}/${batch}/config.json"
            policy_path="${cfg_path}/${env}/${batch}/policy.th"
            output_path="${out_path}/${env}/${batch}/results.npz"

            # Skip if config or policy doesn't exist
            if [[ ! -f "$config_path" || ! -f "$policy_path" ]]; then
                echo "Skipping: Missing files for $model / $env / $batch"
                continue
            fi

            echo "Running test: Model=$model, Env=$env, Batches=$batch"
            python test.py \
                --config "$config_path" \
                --policy "$policy_path" \
                --output "$output_path" \
                --meta-batch-size 20 \
                --num-batches "$batch" \
                --num-workers 8
        done
    done
done

echo "All tests completed."
