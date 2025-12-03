# Modern Model-Agnostic Meta-Learning (MAML) for RL

A comprehensive, modern PyTorch implementation of Model-Agnostic Meta-Learning (MAML) applied to Reinforcement Learning problems.

**Key Contributions & Features:**

* **Multi-Algorithm Support:** Implements **MAML-TRPO** (original), **MAML-PPO** (Proximal Policy Optimization), and **MAML-SAC** (Soft Actor-Critic) as meta-optimizers.
* **Modern Tech Stack:** Fully migrated to **Gymnasium** (replacing the deprecated Gym) and **PyTorch 2.x**.
* **Robust Engineering:** Includes fixes for cross-platform multiprocessing (macOS/Linux compatibility), proper vector environment synchronization, and NumPy 2.0 type safety.
* **Benchmarking:** Supports standard Meta-RL environments: Multi-armed bandits, 2D Navigation, and MuJoCo continuous control (HalfCheetah, Ant).

This repository is built upon the foundational implementation by [Tristan Deleu](https://github.com/tristandeleu/pytorch-maml-rl), extending it with state-of-the-art algorithms and modern software libraries.

## 🛠️ Installation

We recommend using **Miniconda** or **Anaconda** to manage dependencies.

### 1. Create the Environment

```bash
# Create a fresh environment with Python 3.10
conda create -n maml python=3.10 -y

# Activate the environment
conda activate maml
```

### 2. Install Dependencies 
```bash
# Install PyTorch (select the command appropriate for your OS/Hardware)
# For macOS (Apple Silicon/MPS) or Linux (CUDA):
pip install torch torchvision torchaudio

# Install Gymnasium and other requirements
pip install -r requirements.txt
```

`requirements.txt` contents:
```bash
torch>=2.3.0
numpy>=1.20.0
gymnasium[mujoco]>=0.29.0
pyyaml>=6.0
tqdm>=4.66.0
```

## 🚀 Usage
You can train a meta-policy using the train.py script. The script supports selecting the base RL algorithm (TRPO, PPO, or SAC) and the environment configuration.

### General syntax:
```
python train.py --config <config_path> --output-folder <save_path> --model <algorithm> --num-workers <n>
```
