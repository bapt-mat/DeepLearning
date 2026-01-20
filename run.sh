#!/bin/bash
#SBATCH --partition=GPU          # Partition name must be uppercase 'GPU'
#SBATCH --job-name=DeepForg      # Job name
#SBATCH --output=logs/out_%j.log # Standard output log
#SBATCH --error=logs/err_%j.log  # Standard error log
#SBATCH --gres=gpu:1             # Request 1 GPU
#SBATCH --mem=24G                # Request 24GB RAM
#SBATCH --cpus-per-task=4        # Request 4 CPU cores
#SBATCH --time=04:00:00          # Max runtime (4 hours)

# 1. Load Python
# The UJM documentation lists 'Python-3.11.5-ubuntu20' as available.
# We source it directly as recommended in the manual.
echo "🐍 Loading Python 3.11..."
source /home_expes/tools/python/Python-3.11.5-ubuntu20/bin/activate

# 2. Setup Fast Temporary Environment
# We create a virtualenv in the fast local SSD ($SLURM_TMPDIR)
# This prevents filling up your limited Home directory quota.
echo "🔧 Setting up venv in $SLURM_TMPDIR..."
virtualenv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# 3. Install Libraries
# We use --no-cache-dir to save space and ensure a fresh install
echo "📦 Installing dependencies..."
pip install --no-cache-dir torch torchvision opencv-python-headless numpy

# 4. Run Training
# We go to the folder where you submitted the script
cd $SLURM_SUBMIT_DIR

echo "🔥 Starting Training..."
# The data_dir is set to the cluster's shared folder inside train.py, 
# so we don't need to specify it here unless you changed it.
python train.py --epochs 20

echo "✅ Job Finished."