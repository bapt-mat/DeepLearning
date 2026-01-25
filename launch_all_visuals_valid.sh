#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_h5_%j.log
#SBATCH --job-name=VizH5

# 1. SETUP ENV
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

SHARED_VENV="$HOME/DeepForg/venv_shared"
DIRECT_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
INDICES_FILE="best_forged_indices.npy"

if [ -d "$SHARED_VENV" ]; then
    echo "✅ Activating Venv..."
    source /home_expes/tools/python/python3915_0_gpu/bin/activate
    source $SHARED_VENV/bin/activate
else
    echo "❌ Venv not found."
    exit 1
fi

# ==========================================
# PHASE 1: THE ANCHOR (Find the good images)
# ==========================================
ANCHOR_MODEL="segformer_b2_512"

echo "--------------------------------------"
echo "🔍 PHASE 1: Finding Best Forgeries using $ANCHOR_MODEL..."
echo "--------------------------------------"

if [ -f "${ANCHOR_MODEL}.pth" ]; then
    python3 generate_visuals.py \
        --data_dir "$DIRECT_DATA" \
        --save_name "$ANCHOR_MODEL" \
        --arch "segformer" \
        --encoder "mit_b2" \
        --im_size 512 \
        --mode "find" \
        --indices_file "$INDICES_FILE"
else
    echo "❌ Anchor model $ANCHOR_MODEL.pth not found! Cannot find indices."
    exit 1
fi

if [ ! -f "$INDICES_FILE" ]; then
    echo "❌ Failed to generate indices file."
    exit 1
fi

# ==========================================
# PHASE 2: THE FOLLOWERS (Comparison)
# ==========================================
echo "--------------------------------------"
echo "🎨 PHASE 2: Generating comparison visuals..."
echo "--------------------------------------"

# List of other models to compare
MODELS=(
    "unet_baseline"
    "unet_dice"
    "unet_scratch"
    "segformer_b0_baseline"
    "segformer_b2_capacity"
    "segformer_b2_aug"
)

for MODEL in "${MODELS[@]}"; do
    if [ -f "${MODEL}.pth" ]; then
        echo "   Processing $MODEL..."
        
        # Determine Architecture settings
        ARCH="unet"
        ENCODER="resnet34"
        IM_SIZE=256  # Default size for most models
        
        if [[ "$MODEL" == *"deepsup"* ]]; then ARCH="deepsup"; fi
        if [[ "$MODEL" == *"segformer"* ]]; then
            ARCH="segformer"
            ENCODER="mit_b0"
            if [[ "$MODEL" == *"b2"* ]]; then ENCODER="mit_b2"; fi
        fi
        
        # Run in 'use' mode
        python3 generate_visuals_valid.py \
            --data_dir "$DIRECT_DATA" \
            --save_name "$MODEL" \
            --arch "$ARCH" \
            --encoder "$ENCODER" \
            --im_size "$IM_SIZE" \
            --mode "use" \
            --indices_file "$INDICES_FILE"
            
    else
        echo "   ⚠️  Model ${MODEL}.pth not found. Skipping."
    fi
done

echo "✅ All visuals generated for valid indices."