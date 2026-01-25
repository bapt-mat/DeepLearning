#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/segformer_b2_aug_%j.log
#SBATCH --job-name=SegB2Aug

# setup
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

SHARED_VENV="$HOME/DeepForg/venv_shared"
DIRECT_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

# activate shared venv and install albumentations
if [ -d "$SHARED_VENV" ]; then
    echo "Found Shared Venv. Activating..."
    source /home_expes/tools/python/python3915_0_gpu/bin/activate
    source $SHARED_VENV/bin/activate
    
    echo "Checking for albumentations..."
    pip install --no-cache-dir albumentations
else
    echo "Error: Shared Venv not found. Run setup first."
    exit 1
fi

# enable augmentation flag
# This tells dataset.py to use the Albumentations transform
export USE_AUGMENTATION="True"
echo "USE_AUGMENTATION is set to: $USE_AUGMENTATION"

# train 
NAME="segformer_b2_aug"
echo "Training $NAME (SegFormer B2 + Augmentation)..."

python3 train.py \
  --epochs 30 \
  --data_dir "$DIRECT_DATA" \
  --arch segformer \
  --encoder mit_b2 \
  --weights imagenet \
  --loss bce \
  --save_name $NAME

# evaluate
echo "Evaluating $NAME..."
python3 evaluate_full_metrics.py \
  --data_dir "$DIRECT_DATA" \
  --save_name $NAME \
  --arch segformer \
  --encoder mit_b2

echo "Done."