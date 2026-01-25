#!/bin/bash

# ==========================================
# config
# ==========================================
SHARED_VENV="$HOME/DeepForg/venv_shared"

DIRECT_DATA_PATH="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

# ==========================================
# setup jobs
# ==========================================
echo "Generating Setup Script..."

cat <<EOT > setup_env.sh
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=10G
#SBATCH --time=00:45:00
#SBATCH --output=logs/setup_env.log
#SBATCH --job-name=SetupEnv

export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

echo "Setting up Shared Environment at: $SHARED_VENV"

# 1. Load python
source /home_expes/tools/python/python3915_0_gpu/bin/activate

# 2. Create venvv
if [ ! -d "$SHARED_VENV" ]; then
    echo "   Creating new venv..."
    python3 -m venv $SHARED_VENV
else
    echo "   Venv exists. Updating..."
fi

# 3. Install Libraries 
source $SHARED_VENV/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm albumentations scikit-learn pandas numba --extra-index-url https://download.pytorch.org/whl/cu113

echo "Environment Ready."
EOT

# Submit Setup Job
SETUP_JOB_ID=$(sbatch --parsable setup_env.sh)
echo "Submitted Setup Job (ID: $SETUP_JOB_ID)"
echo "The training jobs will wait for Setup to finish..."

# ==========================================
# Submit parallel training jobs
# ==========================================

submit_parallel_job() {
    NAME=$1
    ARCH=$2
    ENCODER=$3
    WEIGHTS=$4
    LOSS=$5
    
    cat <<EOT > run_${NAME}.sh
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/${NAME}_%j.log
#SBATCH --error=logs/${NAME}_err_%j.log
#SBATCH --job-name=$NAME

# 1. SETUP PROXY
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

# 2. ACTIVATE SHARED VENV 
source /home_expes/tools/python/python3915_0_gpu/bin/activate
source $SHARED_VENV/bin/activate

# 3. TRAIN 
echo "Training $NAME (Reading from source)..."
python3 train.py \\
  --epochs 30 \\
  --data_dir "$DIRECT_DATA_PATH" \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --weights $WEIGHTS \\
  --loss $LOSS \\
  --save_name $NAME

# 4. EVALUATE
echo "Evaluating $NAME..."
python3 evaluate_official.py \\
  --data_dir "$DIRECT_DATA_PATH" \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --save_name $NAME

echo "Done."
EOT


    sbatch --dependency=afterany:$SETUP_JOB_ID run_${NAME}.sh
}

echo "Queueing Parallel Jobs..."

# Group A: U-Net
submit_parallel_job "unet_baseline" "unet" "resnet34" "imagenet" "bce"
submit_parallel_job "unet_scratch" "unet" "resnet34" "None" "bce"
submit_parallel_job "unet_dice" "unet" "resnet34" "imagenet" "dice"
submit_parallel_job "unet_deepsup" "deepsup" "resnet34" "imagenet" "bce"

# Group B: SegFormer
submit_parallel_job "segformer_b0_baseline" "segformer" "mit_b0" "imagenet" "bce"
submit_parallel_job "segformer_b2_capacity" "segformer" "mit_b2" "imagenet" "bce"
submit_parallel_job "segformer_b0_scratch" "segformer" "mit_b0" "None" "bce"
submit_parallel_job "segformer_b0_dice" "segformer" "mit_b0" "imagenet" "dice"

echo "----------------------------------------"
echo "All jobs submitted in PARALLEL"
