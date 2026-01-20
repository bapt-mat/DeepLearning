#!/bin/bash
#SBATCH --job-name=ghost_unet
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=03:00:00

# 1. Prepare Environment in TEMP storage (RAM/SSD)
# We do NOT use your Home directory to avoid quota issues
echo "🔧 Setting up environment in $SLURM_TMPDIR..."
module load python/3.11

# Create venv in temp
virtualenv $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate

# Install libs (no-cache to save Home space)
pip install --no-cache-dir torch torchvision opencv-python-headless pandas tqdm matplotlib

# 2. Clone Repo to TEMP storage
cd $SLURM_TMPDIR
echo "⬇️ Cloning repository..."
# CHANGE THIS URL TO YOUR REPO
git clone https://github.com/bapt-mat/DeepLearning.git 
cd DeepLearning

# 3. Stitch Dataset in TEMP storage
echo "🧵 Stitching chunks..."
cat dataset_chunks/part_* > data.zip

# 4. Unzip in TEMP storage
echo "📦 Unzipping..."
unzip -q data.zip -d dataset

# 5. Run Training
echo "🔥 Starting Training..."
python train.py --data_dir "dataset" --epochs 15 --batch_size 16

# 6. SAVE RESULTS (Copy back to Home)
# We copy ONLY the small files (CSV and logs) to avoid quota issues
echo "💾 Saving results..."
HOME_RESULTS="$SLURM_SUBMIT_DIR/results_$SLURM_JOB_ID"
mkdir -p $HOME_RESULTS

# Copy the CSV log
cp training_log.csv $HOME_RESULTS/

# Try to copy the model (might fail if quota is full, but we try)
cp latest_model.pth $HOME_RESULTS/ || echo "⚠️ Could not save model (Quota full?), but logs are saved."

echo "✅ Job Finished. Results in $HOME_RESULTS"