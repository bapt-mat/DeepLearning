#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz_%j.log
#SBATCH --error=logs/viz_err_%j.log
#SBATCH --job-name=GenAllViz

# 1. SETUP
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="$SLURM_TMPDIR"; fi

source /home_expes/tools/python/python3915_0_gpu/bin/activate
if [ ! -d "$TMPDIR/venv" ]; then python3 -m venv $TMPDIR/venv; fi
source $TMPDIR/venv/bin/activate

pip install --no-cache-dir "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 segmentation-models-pytorch --extra-index-url https://download.pytorch.org/whl/cu113

DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

# 2. DEFINE ALL 8 MODELS TO VISUALIZE
# Format: Arch | Encoder | SaveName
MODELS=(
  "unet resnet34 unet_baseline"
  "unet resnet34 unet_scratch"
  "unet resnet34 unet_dice"
  "deepsup resnet34 unet_deepsup"
  "segformer mit_b0 segformer_baseline"
  "segformer mit_b2 segformer_capacity"
  "segformer mit_b0 segformer_scratch"
  "segformer mit_b0 segformer_dice"
)

# 3. LOOP AND GENERATE
for model in "${MODELS[@]}"; do
    set -- $model # Split string into $1, $2, $3
    echo "📸 Generating visuals for $3..."
    
    # Run the generator
    python3 generate_visuals.py --data_dir $DATA --arch $1 --encoder $2 --save_name $3
done

echo "✅ All visuals generated."