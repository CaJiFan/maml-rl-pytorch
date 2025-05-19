#!/bin/bash

# Default values
model="trpo"
cfg_path="configs/maml"

# Get parameters from CLI
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --env) ENV="$2"; shift ;;
        --model) model="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Derived output path
out_path="weights/${model}"

if [[ -z "$ENV" ]]; then
    echo "Please provide an --env parameter (e.g., --env halfcheetah-vel)"
    exit 1
fi

if [[ "$ENV" == "halfcheetah-vel" ]]; then
    for batch in 10 200 500; do
        python train.py --config "${cfg_path}/halfcheetah-vel.yaml" --output-folder "${out_path}/halfcheetah-vel/${batch}" --seed 1 --num-workers 6 --num-batches $batch
    done
elif [[ "$ENV" == "halfcheetah-dir" ]]; then
    for batch in 500; do
        python train.py --config "${cfg_path}/halfcheetah-dir.yaml" --output-folder "${out_path}/halfcheetah-dir/${batch}" --seed 1 --num-workers 6 --num-batches $batch
    done

elif [[ "$ENV" == "2d-navigation" ]]; then
    for batch in 10 200 500; do
        python train.py --config "${cfg_path}/2d-navigation.yaml" --output-folder "${out_path}/2d-navigation/${batch}" --seed 1 --num-workers 6 --num-batches $batch
    done

elif [[ "$ENV" == "bandit-k10-n100" ]]; then
    for batch in 10 200 500; do
        python train.py --config "${cfg_path}/bandit/bandit-k10-n100.yaml" --output-folder "${out_path}/bandit-k10-n100/${batch}" --seed 1 --num-workers 6 --num-batches $batch
    done

elif [[ "$ENV" == "bandit-k5-n100" ]]; then
    for batch in 10 200 500; do
        python train.py --config "${cfg_path}/bandit/bandit-k5-n100.yaml" --output-folder "${out_path}/bandit-k5-n100/${batch}" --seed 1 --num-workers 6 --num-batches $batch
    done

elif [[ "$ENV" == "bandit-k50-n100" ]]; then
    for batch in 10 200 500; do
        python train.py --config "${cfg_path}/bandit/bandit-k50-n100.yaml" --output-folder "${out_path}/bandit-k50-n100/${batch}" --seed 1 --num-workers 6 --num-batches $batch
    done

else
    echo "Unknown env: $ENV"
    exit 1
fi


echo "Training for env='$ENV' with model='$model' complete!"