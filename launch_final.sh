#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=logs/final_pipe_%j.log
#SBATCH --job-name=FinalPipe

# 1. SETUP ENV
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

SHARED_VENV="$HOME/DeepForg/venv_shared"
DIRECT_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

if [ -d "$SHARED_VENV" ]; then
    echo "✅ Found Shared Venv. Activating..."
    source /home_expes/tools/python/python3915_0_gpu/bin/activate
    source $SHARED_VENV/bin/activate
    
    # Install scipy for binary_fill_holes
    pip install --no-cache-dir scipy pandas scikit-learn opencv-python-headless
else
    echo "❌ Error: Shared Venv not found."
    exit 1
fi

# 2. SELECT MODEL
# Ideally, use 'segformer_b2_aug.pth' if it is ready.
# If not, use 'segformer_b2_capacity.pth'.
MODEL="segformer_b2_capacity.pth"

if [ ! -f "$MODEL" ]; then
    echo "⚠️  $MODEL not found, looking for aug version..."
    MODEL="segformer_b2_aug.pth"
fi

# 3. RUN PIPELINE
echo "🔥 Running Final Pipeline with $MODEL..."

python3 predict_final.py \
    --data_dir "$DIRECT_DATA" \
    --model_path "$MODEL" \
    --arch "segformer" \
    --encoder "mit_b2"

echo "✅ Finished."