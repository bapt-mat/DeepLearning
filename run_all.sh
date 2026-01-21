#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=10:00:00
#SBATCH --output=logs/master_%j.log
#SBATCH --job-name=MasterJob

# --- SETUP ---
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="$SLURM_TMPDIR"; fi

source /home_expes/tools/python/python3915_0_gpu/bin/activate
if [ ! -d "$TMPDIR/venv" ]; then python3 -m venv $TMPDIR/venv; fi
source $TMPDIR/venv/bin/activate

echo "📦 Installing..."
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 --extra-index-url https://download.pytorch.org/whl/cu113 --no-cache-dir
pip install --no-cache-dir "numpy<2" h5py opencv-python-headless pandas tqdm scipy segmentation-models-pytorch

DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

# --- 1. TRAINING LOOP (8 Experiments) ---
# Format: Arch | Encoder | Weights | Loss | Name
EXPERIMENTS=(
  "unet resnet34 imagenet bce unet_baseline"
  "unet resnet34 None bce unet_scratch"
  "unet resnet34 imagenet dice unet_dice"
  "deepsup resnet34 None bce unet_deepsup"
  "segformer mit_b0 imagenet bce segformer_baseline"
  "segformer mit_b2 imagenet bce segformer_capacity"
  "segformer mit_b0 None bce segformer_scratch"
  "segformer mit_b0 imagenet dice segformer_dice"
)

for exp in "${EXPERIMENTS[@]}"; do
    set -- $exp # Split string into vars
    echo "🔥 Training $5..."
    python3 train.py --epochs 20 --data_dir $DATA --arch $1 --encoder $2 --weights $3 --loss $4 --save_name $5
    
    echo "📊 Evaluating $5..."
    python3 evaluate_official.py --data_dir $DATA --arch $1 --encoder $2 --save_name $5
done

echo "✅ All 8 models trained and evaluated."