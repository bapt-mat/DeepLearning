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

# --- Stop on errors ---
set -euo pipefail

# --- 1. Setup temp dir ---
export TMPDIR="${SLURM_TMPDIR:-/tmp}"
echo "📂 Using temp dir: $TMPDIR"

# --- 2. Activate GPU Python environment (preinstalled PyTorch + CUDA) ---
echo "🐍 Activating Python 3.10 GPU environment..."
source /home_expes/tools/python/python3102_0_gpu/bin/activate

# --- 3. Set proxy (needed for pip outside cluster) ---
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

# --- 4. Install extra packages if needed ---
pip install --user --no-cache-dir opencv-python-headless pandas tqdm matplotlib

# --- 5. Optional: select specific CUDA version ---
# Example: CUDA 11.0
export PATH=/home_expes/tools/cuda/cuda-11.0/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/home_expes/tools/cuda/cuda-11.0/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

# --- 6. Check PyTorch + GPU availability ---
python3 - <<EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

# --- 7. Run training ---
cd "$SLURM_SUBMIT_DIR" || exit 1
export PYTHONUNBUFFERED=1
echo "🚀 Starting training..."
python3 train.py --epochs 20

echo "✅ Job finished."
