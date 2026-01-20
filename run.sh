#!/bin/bash
#SBATCH --partition=GPU          # Must be 'GPU' for the graphics card [cite: 38]
#SBATCH --gres=gpu:1             # Request 1 GPU [cite: 279]
#SBATCH --mem=24G                # Request 24GB RAM
#SBATCH --time=04:00:00          # 4 hours (adjust if needed)
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
#SBATCH -n 1
#SBATCH -c 4                     # 4 CPUs for data loading
#SBATCH --job-name=DeepForg

# --- 1. Fix the "Not Write-able" Crash ---
# If SLURM doesn't give us a temp folder, force it to use /tmp
if [ -z "$SLURM_TMPDIR" ]; then
    export TMPDIR="/tmp"
else
    export TMPDIR="$SLURM_TMPDIR"
fi
echo "📂 Using temp dir: $TMPDIR"

# --- 2. Load the GPU Python Environment ---
# We use the path you found. This activates the base GPU python.
echo "🐍 Activating University GPU Python..."
source /home_expes/tools/python/python3124_gpu/bin/activate

# --- 3. Create a Writable Layer (Virtual Environment) ---
# The university environment is read-only. We create a fresh venv ON TOP of it
# inside the temp folder so we can install your specific packages (torch, etc.)
echo "🔧 Creating writable venv in temp storage..."
python3 -m venv $TMPDIR/venv --system-site-packages
source $TMPDIR/venv/bin/activate

# --- 4. Install Dependencies ---
# We install exactly what you need into the temp venv
echo "📦 Installing dependencies..."
pip install --no-cache-dir torch torchvision opencv-python-headless numpy

# --- 5. Run Training ---
cd $SLURM_SUBMIT_DIR || exit 1
echo "🔥 Starting Training..."
export PYTHONUNBUFFERED=1
python3 train.py --epochs 20

echo "✅ Job Finished."