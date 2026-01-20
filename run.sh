#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --job-name=DeepForg

# 1. NETWORK PROXY (CRITICAL)
# The docs state this is required to download packages[cite: 88, 89, 90].
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
export http_proxy=http://cache.univ-st-etienne.fr:3128
export https_proxy=http://cache.univ-st-etienne.fr:3128

# 2. SETUP CUDA 11.3
# We explicitly load the CUDA version compatible with your GPU[cite: 80, 81].
export LD_LIBRARY_PATH=/home_expes/tools/cuda/cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/home_expes/tools/cuda/cuda-11.3/bin:$PATH

# 3. SETUP TEMP DIRECTORY
if [ -z "$SLURM_TMPDIR" ]; then
    export TMPDIR="/tmp"
else
    export TMPDIR="$SLURM_TMPDIR"
fi
echo "📂 Using temp dir: $TMPDIR"

# 4. ACTIVATE PYTHON 3.9
# We use Python 3.9 as the base[cite: 51].
echo "🐍 Activating Python 3.9..."
source /home_expes/tools/python/python3915_0_gpu/bin/activate

# 5. CREATE FRESH VENV
# We create a new, empty environment to avoid conflicts.
echo "🔧 Creating fresh venv..."
rm -rf $TMPDIR/venv
python3 -m venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate

# 6. INSTALL COMPATIBLE PYTORCH
# We force install PyTorch 1.12.1 + CUDA 11.3.
# This prevents the "Capability 6.1" error and works with your 1080 Ti.
echo "📦 Downloading compatible PyTorch..."
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir

# 7. INSTALL EXTRAS
echo "📦 Installing libraries..."
pip install --no-cache-dir opencv-python-headless pandas tqdm matplotlib

# 8. RUN TRAINING
cd $SLURM_SUBMIT_DIR || exit 1
echo "🔥 Starting Training..."
export PYTHONUNBUFFERED=1

# Print version info for debugging
python3 -c "import torch; print(f'Torch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}')"

python3 train.py --epochs 20

echo "✅ Job Finished."