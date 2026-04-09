#!/usr/bin/env bash
# Run once on the training machine to get everything ready.
# Tested on Ubuntu 22.04 + CUDA 12.1 + RTX 3060.

set -e

# 1. Clone the G1 MJCF model from MuJoCo Menagerie (only the G1 subfolder)
if [ ! -d "mujoco_menagerie/unitree_g1" ]; then
    echo "Cloning MuJoCo Menagerie (sparse, G1 only)..."
    git clone --filter=blob:none --sparse https://github.com/google-deepmind/mujoco_menagerie.git
    cd mujoco_menagerie
    git sparse-checkout set unitree_g1
    cd ..
else
    echo "mujoco_menagerie/unitree_g1 already present — skipping."
fi

# 2. Create and activate a virtual environment
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv .venv
fi
source .venv/bin/activate

# 3. Install dependencies
echo "Installing Python packages..."
pip install --upgrade pip
pip install -r requirements.txt

# 4. Install PyTorch with CUDA 12.1 (RTX 3060)
pip install torch --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Setup complete."
echo "Activate the environment with:  source .venv/bin/activate"
echo "Then run training with:          python train.py"
