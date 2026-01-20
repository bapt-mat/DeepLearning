#!/bin/bash
#SBATCH --partition=GPU          # [cite: 150, 266]
#SBATCH --gres=gpu:1             # [cite: 391]
#SBATCH --mem=24G                # [cite: 390]
#SBATCH --time=04:00:00          # [cite: 805]
#SBATCH --output=logs/out_%j.log # [cite: 466]
#SBATCH --error=logs/err_%j.log  # [cite: 467]
#SBATCH -n 1                     # [cite: 386]
#SBATCH -c 4                     # [cite: 389]
#SBATCH --job-name=DeepForg      # [cite: 465]

# 1. Setup Temp Directory (Prevent quota issues)
# The docs mention local disks access to /tmp [cite: 169]
if [ -z "$SLURM_TMPDIR" ]; then
    export TMPDIR="/tmp"
else
    export TMPDIR="$SLURM_TMPDIR"
fi
echo "📂 Using temp dir: $TMPDIR"

# 2. Activate University Python 3.9 (GPU)
# We use the specific path found in the UJM doc [cite: 51, 72]
echo "🐍 Activating University GPU Python 3.9..."
source /home_expes/tools/python/python3915_0_gpu/bin/activate

# 3. Create a Writable Layer (Virtual Environment)
# We use --system-site-packages so we can 'see' the University's PyTorch
# This avoids needing to reinstall it (which caused your version mismatch).
echo "🔧 Creating writable venv..."
rm -rf $TMPDIR/venv # Cleanup previous runs
python3 -m venv $TMPDIR/venv --system-site-packages
source $TMPDIR/venv/bin/activate

# 4. Install ONLY Missing Dependencies
# CRITICAL: We do NOT install torch here. We use the one from step 2.
echo "📦 Installing extras..."
pip install --no-cache-dir opencv-python-headless pandas tqdm matplotlib

# 5. Debug Check
# This confirms we are using the correct, cluster-provided PyTorch
echo "🔍 Checking PyTorch version..."
python3 -c "import torch; print(f'Torch: {torch.__version__}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}')"

# 6. Run Training
cd $SLURM_SUBMIT_DIR || exit 1
echo "🔥 Starting Training..."
export PYTHONUNBUFFERED=1
python3 train.py --epochs 20

echo "✅ Job Finished."