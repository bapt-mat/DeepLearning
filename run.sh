#!/bin/bash
#SBATCH --partition=GPU          # We use GPU partition for Deep Learning [cite: 68]
#SBATCH --job-name=DeepForg
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
#SBATCH --gres=gpu:1             # Request 1 GPU [cite: 68]
#SBATCH --mem=24G                # 24GB RAM
#SBATCH --cpus-per-task=4        # 4 CPUs
#SBATCH --time=168:00:00         # Your preferred 7-day limit [cite: 68]

# --- FIX 1: Define a fallback for the temporary directory ---
# If SLURM_TMPDIR is empty (which caused the crash), use /tmp
if [ -z "$SLURM_TMPDIR" ]; then
    export TMPDIR="/tmp"
else
    export TMPDIR="$SLURM_TMPDIR"
fi
echo "📂 Using temp dir: $TMPDIR"

# --- FIX 2: Load Python correctly (Add to PATH, don't source) ---
# We add the 3.11 binary folder to PATH. This works even for raw installs.
export PATH=/home_expes/tools/python/Python-3.11.5-ubuntu20/bin:$PATH
echo "🐍 Python version: $(python3 --version)"

# --- FIX 3: Create a clean virtual environment manually ---
# This avoids the "externally managed environment" error
echo "🔧 Creating venv..."
python3 -m venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate

# --- FIX 4: Install your packages ---
echo "📦 Installing dependencies..."
pip install --no-cache-dir torch torchvision opencv-python-headless numpy

# Run
cd $SLURM_SUBMIT_DIR
echo "🔥 Starting Training..."
python train.py --epochs 20

echo "✅ Job Finished."