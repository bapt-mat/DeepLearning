#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/seg_b2_512_%j.log
#SBATCH --job-name=SegB2_512

# 1. SETUP ENV
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

SHARED_VENV="$HOME/DeepForg/venv_shared"
DIRECT_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

if [ -d "$SHARED_VENV" ]; then
    echo "✅ Found Shared Venv. Activating..."
    source /home_expes/tools/python/python3915_0_gpu/bin/activate
    source $SHARED_VENV/bin/activate
    pip install --no-cache-dir albumentations # Ensure Augmentation lib is there
else
    echo "❌ Error: Shared Venv not found."
    exit 1
fi

# 2. ENABLE AUGMENTATION
export USE_AUGMENTATION="True"

# 3. TRAIN AT 512x512
NAME="segformer_b2_512"
echo "🔥 Training $NAME at 512x512 resolution..."

# Note: Reduced batch_size to 4 to prevent 'Out Of Memory' errors
python3 train.py \
  --epochs 50 \
  --data_dir "$DIRECT_DATA" \
  --arch segformer \
  --encoder mit_b2 \
  --weights imagenet \
  --loss bce \
  --im_size 512 \
  --batch_size 4 \
  --save_name $NAME

# 4. EVALUATE AT 512x512
echo "📊 Evaluating $NAME..."
python3 evaluate_full_metrics.py \
  --data_dir "$DIRECT_DATA" \
  --save_name $NAME \
  --arch segformer \
  --encoder mit_b2 \
  --im_size 512

echo "✅ Done."