#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=04:00:00
#SBATCH --output=logs/long_%j.log
#SBATCH --error=logs/long_err_%j.log
#SBATCH --job-name=LongTrain

# 1. SETUP
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="$SLURM_TMPDIR"; fi

source /home_expes/tools/python/python3915_0_gpu/bin/activate
if [ ! -d "$TMPDIR/venv" ]; then python3 -m venv $TMPDIR/venv; fi
source $TMPDIR/venv/bin/activate

# 2. INSTALL (Safety check)
pip install --no-cache-dir "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 segmentation-models-pytorch numba --extra-index-url https://download.pytorch.org/whl/cu113

SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
LOCAL_DATA="$TMPDIR/dataset"

# 2. COPY DATA (Fast copy using tar)
echo "🚀 Copying data to local SSD..."
mkdir -p $LOCAL_DATA
tar cf - -C $SOURCE_DATA . | tar xf - -C $LOCAL_DATA

# 3. UPDATE DATA PATH IN COMMANDS
# Change $SOURCE_DATA to $LOCAL_DATA in your python commands below
DATA=$LOCAL_DATA

# 4. RUN LONG TRAINING (60 Epochs)
echo "🔥 Starting Long Training for U-Net Baseline..."

python3 train.py \
  --epochs 60 \
  --data_dir $DATA \
  --arch unet \
  --encoder resnet34 \
  --weights imagenet \
  --loss bce \
  --save_name unet_baseline_long

echo "📊 Evaluating the Long Model..."
python3 evaluate_official.py \
  --data_dir $DATA \
  --arch unet \
  --encoder resnet34 \
  --save_name unet_baseline_long

echo "✅ Job Finished."