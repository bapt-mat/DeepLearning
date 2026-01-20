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
echo "📂 Using temp dir: $TMPDIR"

# 4. ACTIVATE PYTHON 3.9
echo "🐍 Activating Python 3.9..."
source /home_expes/tools/python/python3915_0_gpu/bin/activate

# 5. CREATE FRESH VENV
echo "🔧 Creating fresh venv..."
rm -rf $TMPDIR/venv
python3 -m venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate

# 6. INSTALL PYTORCH (GTX 1080 Ti Compatible)
echo "📦 Downloading compatible PyTorch..."
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir

# 7. INSTALL EXTRAS (Fixing NumPy crash here!)
echo "📦 Installing libraries (forcing numpy<2)..."
# We force "numpy<2" because PyTorch 1.12 crashes with NumPy 2.0
pip install --no-cache-dir "numpy<2" opencv-python-headless pandas tqdm matplotlib

# 8. RUN TRAINING
cd $SLURM_SUBMIT_DIR || exit 1
echo "🔥 Starting Training..."
export PYTHONUNBUFFERED=1

# (Fixed the quoting syntax error below)
python3 -c "import torch; print(f'Torch: {torch.__version__} | CUDA: {torch.version.cuda} | GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"

python3 train.py --epochs 10

echo "✅ Job Finished."