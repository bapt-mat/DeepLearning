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

# 1. NETWORK PROXY
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

# 5. CREATE FRESH VENV (Use the one from training if it exists, or create new)
# It's faster to reuse the one created by run.sh if you didn't delete it.
# If it's gone, we recreate it:
if [ ! -d "$TMPDIR/venv" ]; then
    python3 -m venv $TMPDIR/venv
    source $TMPDIR/venv/bin/activate
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir
    pip install --no-cache-dir "numpy<2" opencv-python-headless pandas tqdm matplotlib segmentation-models-pytorch numba scipy
else
    source $TMPDIR/venv/bin/activate
fi

# 6. RUN EVALUATION
echo "🚀 Starting Evaluation..."
python3 evaluate_official.py

echo "Evaluation Finished."