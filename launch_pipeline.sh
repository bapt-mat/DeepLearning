#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pipeline_safe_%j.log
#SBATCH --job-name=PipeSafe

# 1. SETUP ENV
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

SHARED_VENV="$HOME/DeepForg/venv_shared"
# POINT DIRECTLY TO NETWORK STORAGE (Zero Disk Usage)
DIRECT_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

# 2. ACTIVATE SHARED VENV & INSTALL LIBS
if [ -d "$SHARED_VENV" ]; then
    echo "✅ Found Shared Venv. Activating..."
    source /home_expes/tools/python/python3915_0_gpu/bin/activate
    source $SHARED_VENV/bin/activate
    
    # INSTALL MISSING LIBRARIES (Fixes "No module named pandas/sklearn")
    echo "📦 Checking for pandas & scikit-learn..."
    pip install --no-cache-dir pandas scikit-learn
else
    echo "❌ Error: Shared Venv not found. Run setup first."
    exit 1
fi

# 3. RUN PIPELINE
echo "🔥 Running Two-Stage Pipeline Evaluation..."

# CLASSIFIER = segformer_b2_capacity (The Gatekeeper)
# SEGMENTER  = unet_baseline       (The Specialist)

if [ -f "segformer_b2_capacity.pth" ] && [ -f "unet_baseline.pth" ]; then
    python3 evaluate_pipeline.py \
        --data_dir "$DIRECT_DATA" \
        --cls_model "segformer_b2_capacity" \
        --cls_arch "segformer" \
        --cls_enc "mit_b2" \
        --seg_model "unet_baseline" \
        --seg_arch "unet" \
        --seg_enc "resnet34"
else
    echo "❌ Error: One or both model weights (.pth) are missing!"
    echo "   Ensure 'segformer_b2_capacity.pth' and 'unet_baseline.pth' are in this folder."
fi

echo "✅ Pipeline evaluation finished."