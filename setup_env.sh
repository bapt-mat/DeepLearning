#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --job-name=SetupEnv

# 1. SETUP
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

# Define a shared location for the environment (e.g., in your home or scratch)
# We use $HOME to ensure it persists and has space.
VENV_PATH="$HOME/venv_master_100"

echo "🔧 Creating Master Environment at $VENV_PATH..."
source /home_expes/tools/python/python3915_0_gpu/bin/activate
python3 -m venv $VENV_PATH

# 2. INSTALL (Only Once)
source $VENV_PATH/bin/activate
pip install --no-cache-dir --upgrade pip
# Installing all libraries needed for ALL experiments
pip install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm albumentations scikit-learn pandas numba --extra-index-url https://download.pytorch.org/whl/cu113

echo "✅ Master Environment Created."