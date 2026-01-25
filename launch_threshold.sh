#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=01:00:00
#SBATCH --output=logs/thresh_safe_%j.log
#SBATCH --job-name=ThreshSafe

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
    
    # INSTALL MISSING LIBRARIES (Fixes "No module named pandas")
    echo "Checking for pandas & matplotlib..."
    pip install --no-cache-dir pandas matplotlib scikit-learn
else
    echo "Error: Shared Venv not found. Run setup first."
    exit 1
fi

# run threshold study
# We run this the best model: segformer_b2_capacity
MODEL_PATH="segformer_b2_capacity.pth"

if [ -f "$MODEL_PATH" ]; then
    echo "Running Threshold Study on $MODEL_PATH..."
    
    python3 study_threshold.py \
        --data_dir "$DIRECT_DATA" \
        --model_path "$MODEL_PATH" \
        --arch "segformer" \
        --encoder "mit_b2"
else
    echo "Model file '$MODEL_PATH' not found in current directory!"
fi

echo "Threshold study complete."