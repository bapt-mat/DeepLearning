#!/bin/bash
#SBATCH --job-name=DeepForg_Train
#SBATCH --output=logs/out_%j.log
#SBATCH --error=logs/err_%j.log
#SBATCH --partition=gpu          # <--- Critical: Use the GPU partition
#SBATCH --gres=gpu:1             # <--- Critical: Request 1 GPU
#SBATCH --cpus-per-task=4        # 4 CPUs is plenty for dataloading
#SBATCH --mem=24G                # 24GB RAM is sufficient
#SBATCH --time=04:00:00          # 4 hours is enough for 10-20 epochs

# 1. Load the correct Python module (as we tested before)
module load python/3.11

# 2. Setup a fast temporary environment (RAM/SSD)
# We do this to avoid filling up your Home directory quota
echo "🔧 Setting up virtual environment..."
export ENV_DIR=$SLURM_TMPDIR/venv
virtualenv $ENV_DIR
source $ENV_DIR/bin/activate

# 3. Install dependencies
# --no-cache-dir is CRITICAL to save space
pip install --no-cache-dir torch torchvision opencv-python-headless numpy

# 4. Run Training
echo "🔥 Starting Training..."
# Ensure we are in the right folder
cd ~/DeepForg || exit 1
export PYTHONUNBUFFERED=1

# Run the training script
python train.py --epochs 20

echo "✅ Job Finished."