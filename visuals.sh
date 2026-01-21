#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_%j.log
#SBATCH --error=logs/viz_err_%j.log
#SBATCH --job-name=GenViz

# --- CONFIGURATION ---
# Change this to the model you want to visualize (must match the .pth filename without extension)
MODEL_NAME="unet_baseline_long" 
ARCH="unet"
ENCODER="resnet34"

# 1. SETUP PROXY
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="$SLURM_TMPDIR"; fi

# 2. ACTIVATE ENVIRONMENT
source /home_expes/tools/python/python3915_0_gpu/bin/activate
if [ ! -d "$TMPDIR/venv" ]; then python3 -m venv $TMPDIR/venv; fi
source $TMPDIR/venv/bin/activate

# 3. ENSURE DEPENDENCIES
pip install --no-cache-dir "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 segmentation-models-pytorch --extra-index-url https://download.pytorch.org/whl/cu113

# 4. DEFINE DATA PATH
DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

# 5. RUN GENERATOR
echo "📸 Generating visuals for $MODEL_NAME..."

python3 generate_visuals.py \
  --data_dir $DATA \
  --save_name $MODEL_NAME \
  --arch $ARCH \
  --encoder $ENCODER

echo "✅ Job Finished. Output saved to visuals_${MODEL_NAME}.h5"