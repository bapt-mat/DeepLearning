#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:45:00
#SBATCH --output=logs/study_area_256_%j.log
#SBATCH --job-name=AreaStudy256

# seetup
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

SHARED_VENV="$HOME/DeepForg/venv_shared"
DIRECT_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

if [ -d "$SHARED_VENV" ]; then
    echo "Found Shared Venv. Activating..."
    source /home_expes/tools/python/python3915_0_gpu/bin/activate
    source $SHARED_VENV/bin/activate
    pip install --no-cache-dir pandas matplotlib scipy
else
    echo "Error: Shared Venv not found."
    exit 1
fi

# config
MODEL="segformer_b2_capacity.pth"
SIZE=256

if [ ! -f "$MODEL" ]; then
    echo "Error: Model '$MODEL' not found!"
    exit 1
fi

# run study
echo "Running Minimum Area Study on $MODEL ($SIZE px)..."

python3 study_min_area.py \
    --data_dir "$DIRECT_DATA" \
    --model_path "$MODEL" \
    --arch "segformer" \
    --encoder "mit_b2" \
    --im_size $SIZE

echo "Done."