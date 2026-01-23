#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=logs/setup_env.log
#SBATCH --job-name=SetupEnv

# 1. SETUP
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

# Define Shared Venv Path (In your home folder for persistence)
# We use $HOME so all jobs can find it later
SHARED_VENV="$HOME/DeepForg/venv_shared"

echo "🔧 Building Shared Environment at: $SHARED_VENV"

# 2. CREATE VENV
source /home_expes/tools/python/python3915_0_gpu/bin/activate
python3 -m venv $SHARED_VENV

# 3. INSTALL LIBRARIES (ONCE FOR EVERYONE)
source $SHARED_VENV/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm albumentations scikit-learn pandas numba --extra-index-url https://download.pytorch.org/whl/cu113

echo "✅ Shared Environment Ready."