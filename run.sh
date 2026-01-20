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

# 2. Activate OLDER GPU Python (Python 3.10)
# python3124 (Py3.12) was too new. python3102 (Py3.10) supports GTX 1080 Ti.
echo "🐍 Activating Python 3.10 (Compatible with GTX 1080 Ti)..."
source /home_expes/tools/python/python3102_0_gpu/bin/activate

# 3. Clean and Create Venv
# We MUST delete the old venv because it was created with Python 3.12
# and will break if we try to use it with Python 3.10.
echo "🧹 Cleaning old venv..."
rm -rf $TMPDIR/venv

echo "🔧 Creating new writable venv..."
python3 -m venv $TMPDIR/venv --system-site-packages
source $TMPDIR/venv/bin/activate

# 4. Install Dependencies
# Still excluding torch/numpy so we use the cluster's compatible versions
echo "📦 Installing extras..."
pip install --no-cache-dir opencv-python-headless pandas tqdm matplotlib

# 5. Check Versions (For peace of mind)
echo "🔍 Checking PyTorch compatibility..."
python3 -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}')"

# 6. Run Training
cd $SLURM_SUBMIT_DIR || exit 1
echo "🔥 Starting Training..."
export PYTHONUNBUFFERED=1
python3 train.py --epochs 20

echo "✅ Job Finished."