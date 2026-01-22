#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/unet_aug_%j.log
#SBATCH --job-name=UnetAug

# 1. SETUP ENV
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="$SLURM_TMPDIR"; fi

source /home_expes/tools/python/python3915_0_gpu/bin/activate

# Create specific venv
VENV_PATH="$TMPDIR/venv_unet_aug"
python3 -m venv $VENV_PATH
PYBIN="$VENV_PATH/bin/python3"
PIP="$VENV_PATH/bin/pip"

# 2. INSTALL LIBRARIES + ALBUMENTATIONS
echo "📦 Installing libraries..."
$PIP install --no-cache-dir --upgrade pip
$PIP install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm albumentations scikit-learn pandas --extra-index-url https://download.pytorch.org/whl/cu113

# 3. DATA TRANSFER
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
LOCAL_DATA="$TMPDIR/dataset_aug" 
echo "🚀 Unpacking data..."
mkdir -p $LOCAL_DATA
tar cf - -C $SOURCE_DATA . | tar xf - -C $LOCAL_DATA

# 4. ENABLE AUGMENTATION FLAG
export USE_AUGMENTATION="True"
echo "🌪️  USE_AUGMENTATION is set to: $USE_AUGMENTATION"

# 5. TRAIN
NAME="unet_aug"
echo "🔥 Training $NAME..."

$PYBIN train.py \
  --epochs 100 \
  --data_dir $LOCAL_DATA \
  --arch unet \
  --encoder resnet34 \
  --weights imagenet \
  --loss bce \
  --save_name $NAME

# 6. EVALUATE
echo "📊 Evaluating $NAME..."
$PYBIN evaluate_full_metrics.py \
  --data_dir $LOCAL_DATA \
  --save_name $NAME \
  --arch unet \
  --encoder resnet34

# Cleanup
rm -rf $LOCAL_DATA
rm -rf $VENV_PATH
echo "✅ Done."