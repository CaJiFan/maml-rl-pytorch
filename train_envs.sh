#!/bin/bash

#python train.py --config configs/maml/halfcheetah-vel.yaml --output-folder weights/ppo/maml-halfcheetah-vel --seed 1 --num-workers 8
# python train.py --config configs/maml/halfcheetah-dir.yaml --output-folder weights/ppo/maml-halfcheetah-dir --seed 1 --num-workers 8
python train.py --config configs/maml/2d-navigation.yaml --output-folder weights/trpo/2d-navigation/10 --seed 1 --num-workers 8
#python train.py --config configs/maml/ant-dir.yaml --output-folder weights/ant-dir --seed 1 --num-workers 8
#python train.py --config configs/maml/ant-goal.yaml --output-folder weights/ant-goal --seed 1 --num-workers 8
#python train.py --config configs/maml/ant-vel.yaml --output-folder weights/ant-vel --seed 1 --num-workers 8
# python train.py --config configs/maml/bandit/bandit-k10-n100.yaml --output-folder weights/ppo/bandit-k10-n100 --seed 1 --num-workers 8 
# python train.py --config configs/maml/bandit/bandit-k50-n100.yaml --output-folder weights/ppo/bandit-k50-n100 --seed 1 --num-workers 8        
# python train.py --config configs/maml/bandit/bandit-k5-n100.yaml --output-folder weights/ppo/bandit-k5-n100 --seed 1 --num-workers 8

echo "All envs done!..."
