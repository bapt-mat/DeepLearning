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

# 2. Activate University GPU Python
# This environment ALREADY HAS a compatible PyTorch installed!
echo "🐍 Activating University GPU Python..."
source /home_expes/tools/python/python3124_gpu/bin/activate

# 3. Create a Writable Layer
# We use --system-site-packages so we can "see" the University's PyTorch
# without needing to reinstall it.
echo "🔧 Creating venv..."
python3 -m venv $TMPDIR/venv --system-site-packages
source $TMPDIR/venv/bin/activate

# 4. Install ONLY Missing Dependencies
# 🛑 CRITICAL: Do NOT include 'torch', 'torchvision', or 'numpy' here.
# We only install the extras that the cluster usually lacks.
echo "📦 Installing extras..."
pip install --no-cache-dir opencv-python-headless pandas tqdm matplotlib

# 5. Debug Check (Optional)
# This prints which Torch version is being used to the logs
echo "🔍 Checking PyTorch version..."
python3 -c "import torch; print(f'Torch: {torch.__version__}, CUDA: {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 6. Run Training
cd $SLURM_SUBMIT_DIR || exit 1
echo "🔥 Starting Training..."
export PYTHONUNBUFFERED=1
python3 train.py --epochs 20

echo "✅ Job Finished."