#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/viz_gen_%j.log
#SBATCH --job-name=VizGen

# 1. SETUP ENV
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="$SLURM_TMPDIR"; fi

# Setup Python (Same as training)
source /home_expes/tools/python/python3915_0_gpu/bin/activate
# Create/Activate venv if needed (using the same one from launch_all_fixed is safest)
# Or create a quick temp one:
python3 -m venv $TMPDIR/venv
source $TMPDIR/venv/bin/activate
pip install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm --extra-index-url https://download.pytorch.org/whl/cu113

# 2. DATA TRANSFER
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
LOCAL_DATA="$TMPDIR/dataset_viz"
mkdir -p $LOCAL_DATA
tar cf - -C $SOURCE_DATA . | tar xf - -C $LOCAL_DATA

# 3. GENERATE VISUALS FOR KEY MODELS
# Add any model you want to visualize here
MODELS=("unet_baseline" "segformer_b0_baseline" "unet_baseline_long")

for MODEL in "${MODELS[@]}"; do
    if [ -f "${MODEL}.pth" ]; then
        echo "📸 Generating visuals for $MODEL..."
        python3 generate_visuals.py \
            --data_dir $LOCAL_DATA \
            --save_name $MODEL \
            --arch $( [[ "$MODEL" == *"segformer"* ]] && echo "segformer" || echo "unet" ) \
            --encoder $( [[ "$MODEL" == *"segformer"* ]] && echo "mit_b0" || echo "resnet34" )
    else
        echo "⚠️  Model ${MODEL}.pth not found. Skipping."
    fi
done

echo "✅ All visuals generated."