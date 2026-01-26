#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/eval_safe_%j.log
#SBATCH --job-name=Eval

# setup
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

SHARED_VENV="$HOME/DeepForg/venv_shared"
DIRECT_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

# activate shared venv 
if [ -d "$SHARED_VENV" ]; then
    echo "Found Shared Venv. Activating..."
    source /home_expes/tools/python/python3915_0_gpu/bin/activate
    source $SHARED_VENV/bin/activate
    
    echo "Checking for pandas and scikit-learn"
    pip install --no-cache-dir pandas scikit-learn
else
    echo "Error: Shared Venv not found. Run setup first."
    exit 1
fi

# list models
MODELS=(
    "unet_baseline"
    "unet_dice"
    "unet_scratch"
    "unet_deepsup"
    "segformer_b0_baseline"
    "segformer_b0_dice"
    "segformer_b0_scratch"
    "segformer_b2_capacity"
    "segformer_b2_aug"
)

# evaluation loop
for MODEL in "${MODELS[@]}"; do
    if [ -f "${MODEL}.pth" ]; then
        echo "Evaluating $MODEL..."
        
        # Configure Arch/Encoder
        ARCH="unet"
        ENCODER="resnet34"
        if [[ "$MODEL" == *"deepsup"* ]]; then ARCH="deepsup"; fi
        if [[ "$MODEL" == *"segformer"* ]]; then 
            ARCH="segformer"
            ENCODER="mit_b0"
            if [[ "$MODEL" == *"b2"* ]]; then ENCODER="mit_b2"; fi
        fi
        
        python3 evaluate_full_metrics.py \
            --data_dir "$DIRECT_DATA" \
            --save_name "$MODEL" \
            --arch "$ARCH" \
            --encoder "$ENCODER"
    else
        echo "${MODEL}.pth not found. Skipping."
    fi
done

echo "All evaluations complete."