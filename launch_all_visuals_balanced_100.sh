#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=00:20:00
#SBATCH --output=logs/viz_h5_%j.log
#SBATCH --job-name=VizH5

export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

SHARED_VENV="$HOME/DeepForg/venv_shared"
DIRECT_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

if [ -d "$SHARED_VENV" ]; then
    echo "Found Shared Venv. Activating..."
    source /home_expes/tools/python/python3915_0_gpu/bin/activate
    source $SHARED_VENV/bin/activate
else
    echo "Error: Shared Venv not found. Run setup first."
    exit 1
fi

#models
MODELS=(
    "unet_baseline_100"
    "unet_dice_100"
    "unet_scratch_100"
    "unet_deepsup_100"
    "segformer_b0_baseline_100"
    "segformer_b0_dice_100"
    "segformer_b0_scratch_100"
    "segformer_b2_capacity_100"
    "segformer_b2_aug_100"
)

for MODEL in "${MODELS[@]}"; do
    if [ -f "${MODEL}.pth" ]; then
        echo "Processing $MODEL..."
        
        ARCH="unet"
        ENCODER="resnet34"
        
        if [[ "$MODEL" == *"deepsup"* ]]; then ARCH="deepsup"; fi
        if [[ "$MODEL" == *"segformer"* ]]; then
            ARCH="segformer"
            ENCODER="mit_b0"
            if [[ "$MODEL" == *"b2"* ]]; then ENCODER="mit_b2"; fi
        fi
        
        python3 generate_balanced.py \
            --data_dir "$DIRECT_DATA" \
            --save_name "$MODEL" \
            --arch "$ARCH" \
            --encoder "$ENCODER"
    else
        echo "Model ${MODEL}.pth not found. Skipping."
    fi
done

echo "All H5 files created."