#!/bin/bash

model="trpo"
cfg_path="weights/${model}"
out_path="results/${model}"

# python test.py --config "${cfg_path}/maml-halfcheetah-vel/config.json" --policy "${cfg_path}/maml-halfcheetah-vel/policy.th" --output "${out_path}/maml-halfcheetah-vel/results.npz" --meta-batch-size 20 --num-batches 10  --num-workers 8
# python test.py --config "${cfg_path}/maml-halfcheetah-dir/config.json" --policy "${cfg_path}/maml-halfcheetah-dir/policy.th" --output "${out_path}/maml-halfcheetah-dir/results.npz" --meta-batch-size 20 --num-batches 10  --num-workers 8
python test.py --config "${cfg_path}/2d-navigation/10/config.json" --policy "${cfg_path}/2d-navigation/10/policy.th" --output "${out_path}/2d-navigation/10/results.npz" --meta-batch-size 20 --num-batches 10  --num-workers 8
python test.py --config "${cfg_path}/2d-navigation/200/config.json" --policy "${cfg_path}/2d-navigation/200/policy.th" --output "${out_path}/2d-navigation/200/results.npz" --meta-batch-size 20 --num-batches 10  --num-workers 8
python test.py --config "${cfg_path}/2d-navigation/500/config.json" --policy "${cfg_path}/2d-navigation/500/policy.th" --output "${out_path}/2d-navigation/500/results.npz" --meta-batch-size 20 --num-batches 10  --num-workers 8
# python test.py --config "${cfg_path}/bandit-k10-n100/config.json" --policy "${cfg_path}/bandit-k10-n100/policy.th" --output "${out_path}/bandit-k10-n100/results.npz" --meta-batch-size 20 --num-batches 10  --num-workers 8
# python test.py --config "${cfg_path}/bandit-k50-n100/config.json" --policy "${cfg_path}/bandit-k50-n100/policy.th" --output "${out_path}/bandit-k50-n100/results.npz" --meta-batch-size 20 --num-batches 10  --num-workers 8
# python test.py --config "${cfg_path}/bandit-k5-n100/config.json" --policy "${cfg_path}/bandit-k5-n100/policy.th" --output "${out_path}/bandit-k5-n100/results.npz" --meta-batch-size 20 --num-batches 10  --num-workers 8
