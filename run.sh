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

# --- Fail fast if any command fails ---
set -euo pipefail

# --- 1. Setup temp dir ---
if [ -z "${SLURM_TMPDIR:-}" ]; then
    export TMPDIR="/tmp"
else
    export TMPDIR="$SLURM_TMPDIR"
fi
echo "📂 Using temp dir: $TMPDIR"

# --- 2. Activate GPU Python (Python 3.10 compatible with GTX 1080 Ti) ---
echo "🐍 Activating Python 3.10 GPU environment..."
source /home_expes/tools/python/python3102_0_gpu/bin/activate

# --- 3. Install extra packages if needed ---
# Use --user to avoid touching system directories
echo "📦 Installing Python extras..."
pip install --user --no-cache-dir opencv-python-headless pandas tqdm matplotlib

# --- 4. Check PyTorch availability ---
python3 - <<EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

# --- 5. Run training ---
cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONUNBUFFERED=1

echo "🚀 Starting training..."
python3 train.py --epochs 20

echo "✅ Job finished."
