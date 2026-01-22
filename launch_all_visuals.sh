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

# Setup Python
source /home_expes/tools/python/python3915_0_gpu/bin/activate

# Try to reuse an existing venv to save time, or create a new one
if [ -d "venv_segformer_b0_baseline" ]; then
    source venv_segformer_b0_baseline/bin/activate
elif [ -d "venv_unet_baseline" ]; then
    source venv_unet_baseline/bin/activate
else
    # Fallback: Create temp venv
    python3 -m venv $TMPDIR/venv
    source $TMPDIR/venv/bin/activate
    pip install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm --extra-index-url https://download.pytorch.org/whl/cu113
fi

# 2. DATA TRANSFER
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
LOCAL_DATA="$TMPDIR/dataset_viz"
mkdir -p $LOCAL_DATA
tar cf - -C $SOURCE_DATA . | tar xf - -C $LOCAL_DATA

# 3. LIST OF ALL MODELS TO PROCESS
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
        
        # --- SMART CONFIGURATION ---
        # 1. Default U-Net settings
        ARCH="unet"
        ENCODER="resnet34"
        
        # 2. Check for Deep Supervision
        if [[ "$MODEL" == *"deepsup"* ]]; then
            ARCH="deepsup"
            ENCODER="resnet34" # Dummy value, ignored by deepsup model
        fi
        
        # 3. Check for SegFormer
        if [[ "$MODEL" == *"segformer"* ]]; then
            ARCH="segformer"
            ENCODER="mit_b0" # Default B0
            
            # Check for B2 Capacity model
            if [[ "$MODEL" == *"b2"* ]]; then
                ENCODER="mit_b2"
            fi
        fi
        
        python3 generate_visuals.py \
            --data_dir $LOCAL_DATA \
            --save_name $MODEL \
            --arch $ARCH \
            --encoder $ENCODER
    else
        echo "⚠️  Model ${MODEL}.pth not found. Skipping."
    fi
done

echo "✅ All visuals generated."