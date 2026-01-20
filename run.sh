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

# 1. Setup Temp Directory
if [ -z "$SLURM_TMPDIR" ]; then
    export TMPDIR="/tmp"
else
    export TMPDIR="$SLURM_TMPDIR"
fi
echo "📂 Using temp dir: $TMPDIR"

# 2. Activate Python 3.9 GPU
# Python 3.9 is extremely stable for ML and usually comes pre-loaded.
echo "🐍 Activating Python 3.9 (Stable GPU)..."
source /home_expes/tools/python/python3915_0_gpu/bin/activate

# 3. Clean and Create Venv
echo "🧹 Cleaning old venv..."
rm -rf $TMPDIR/venv

echo "🔧 Creating new writable venv..."
# --system-site-packages allows us to see the pre-installed Torch
python3 -m venv $TMPDIR/venv --system-site-packages
source $TMPDIR/venv/bin/activate

# 4. Install Dependencies
# We assume the cluster HAS torch. If this fails again, we will force install an old version.
echo "📦 Installing extras..."
pip install --no-cache-dir opencv-python-headless pandas tqdm matplotlib

# 5. Debug Check
echo "🔍 Checking PyTorch..."
python3 -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

# 6. Run Training
cd $SLURM_SUBMIT_DIR || exit 1
echo "🔥 Starting Training..."
export PYTHONUNBUFFERED=1
python3 train.py --epochs 20

echo "✅ Job Finished."