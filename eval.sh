#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_out_%j.log
#SBATCH --error=logs/eval_err_%j.log
#SBATCH -n 1
#SBATCH -c 4
#SBATCH --job-name=EvalForg

# 1. NETWORK PROXY (Essential for downloads)
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
export http_proxy=http://cache.univ-st-etienne.fr:3128
export https_proxy=http://cache.univ-st-etienne.fr:3128

# 2. SETUP CUDA 11.3
export LD_LIBRARY_PATH=/home_expes/tools/cuda/cuda-11.3/lib64:$LD_LIBRARY_PATH
export PATH=/home_expes/tools/cuda/cuda-11.3/bin:$PATH

# 3. SETUP TEMP DIR
if [ -z "$SLURM_TMPDIR" ]; then
    export TMPDIR="/tmp"
else
    export TMPDIR="$SLURM_TMPDIR"
fi

# 4. ACTIVATE PYTHON 3.9
source /home_expes/tools/python/python3915_0_gpu/bin/activate

# 5. SETUP VENV & FORCE INSTALL DEPENDENCIES
# We create the venv if missing, but we ALWAYS run pip install
# to ensure 'scipy' is present.
if [ ! -d "$TMPDIR/venv" ]; then
    echo "🔧 Creating new venv..."
    python3 -m venv $TMPDIR/venv
fi

source $TMPDIR/venv/bin/activate

echo "📦 Verifying dependencies (installing missing ones)..."
# We force installation of scipy and numba here
pip install --no-cache-dir "numpy<2" scipy numba pandas tqdm opencv-python-headless segmentation-models-pytorch torch==1.12.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113

# 6. RUN EVALUATION
echo "🚀 Starting Evaluation..."
python3 evaluate_official.py

echo "✅ Evaluation Finished."