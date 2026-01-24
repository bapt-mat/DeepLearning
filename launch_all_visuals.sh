#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=00:30:00
#SBATCH --output=logs/viz_safe_%j.log
#SBATCH --job-name=VizSafe

# 1. SETUP ENV
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

# Define paths
SHARED_VENV="$HOME/DeepForg/venv_shared"
# POINT DIRECTLY TO NETWORK STORAGE (No Copying!)
DIRECT_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

# 2. ACTIVATE SHARED VENV & INSTALL MATPLOTLIB
if [ -d "$SHARED_VENV" ]; then
    echo "✅ Found Shared Venv. Activating..."
    source /home_expes/tools/python/python3915_0_gpu/bin/activate
    source $SHARED_VENV/bin/activate
    
    # INSTALL MATPLOTLIB (Fixes "No module named matplotlib")
    echo "📦 Checking for matplotlib..."
    pip install --no-cache-dir matplotlib
else
    echo "❌ Error: Shared Venv not found at $SHARED_VENV"
    echo "   Please run the setup script (00_setup_env.sh) first."
    exit 1
fi

# 3. LIST OF MODELS TO VISUALIZE
# (Ensure these .pth files exist in your current folder)
MODELS=(
    "unet_baseline"
    "unet_dice"
    "unet_scratch"
    "unet_deepsup"
    "segformer_b0_baseline"
    "segformer_b0_dice"
    "segformer_b0_scratch"
    "segformer_b2_capacity"
)

# 4. GENERATE VISUALS LOOP
for MODEL in "${MODELS[@]}"; do
    if [ -f "${MODEL}.pth" ]; then
        echo "📸 Generating visuals for $MODEL..."
        
        # Smart Architecture Detection
        ARCH="unet"
        ENCODER="resnet34"
        
        if [[ "$MODEL" == *"deepsup"* ]]; then
            ARCH="deepsup"
        fi
        
        if [[ "$MODEL" == *"segformer"* ]]; then
            ARCH="segformer"
            ENCODER="mit_b0"
            if [[ "$MODEL" == *"b2"* ]]; then
                ENCODER="mit_b2"
            fi
        fi
        
        # RUN PYTHON SCRIPT (Reading directly from Source)
        python3 generate_visuals.py \
            --data_dir "$DIRECT_DATA" \
            --save_name "$MODEL" \
            --arch "$ARCH" \
            --encoder "$ENCODER"
            
    else
        echo "⚠️  Model ${MODEL}.pth not found. Skipping."
    fi
done

echo "✅ All visuals generated in 'visuals/' folder."