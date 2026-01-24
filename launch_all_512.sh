#!/bin/bash

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# 1. Path to the SHARED Environment (Built once, used by all)
SHARED_VENV="$HOME/DeepForg/venv_shared"

# 2. Path to the ORIGINAL DATASET (Read directly, NO COPYING)
DIRECT_DATA_PATH="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

# ==========================================
# 🛠️ STEP 1: SETUP JOBS (Build Venv Once)
# ==========================================
echo "📦 Generating Setup Script..."

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

echo "🔧 Setting up Shared Environment at: $SHARED_VENV"

# 1. Load Base Python
source /home_expes/tools/python/python3915_0_gpu/bin/activate

# 2. Create Venv (if not exists)
if [ ! -d "$SHARED_VENV" ]; then
    echo "   Creating new venv..."
    python3 -m venv $SHARED_VENV
else
    echo "   Venv exists. Updating..."
fi

# 3. Install Libraries (Added 'numba' to fix your previous error)
source $SHARED_VENV/bin/activate
pip install --no-cache-dir --upgrade pip
pip install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm albumentations scikit-learn pandas numba --extra-index-url https://download.pytorch.org/whl/cu113

echo "✅ Environment Ready."
EOT

# Submit Setup Job
SETUP_JOB_ID=$(sbatch --parsable setup_env.sh)
echo "🚀 Submitted Setup Job (ID: $SETUP_JOB_ID)"
echo "⏳ The training jobs will wait for Setup to finish..."

# ==========================================
# 🚀 STEP 2: SUBMIT PARALLEL TRAINING JOBS
# ==========================================
# We submit all 8 jobs NOW, but they will wait (dependency) for the Setup to finish.
# Once Setup is done, they will ALL start efficiently.

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
#SBATCH --time=24:00:00
#SBATCH --output=logs/${NAME}_%j.log
#SBATCH --error=logs/${NAME}_err_%j.log
#SBATCH --job-name=$NAME

# 1. SETUP PROXY
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

# 2. ACTIVATE SHARED VENV (Fixes 'libpython' error too)
source /home_expes/tools/python/python3915_0_gpu/bin/activate
source $SHARED_VENV/bin/activate

# 3. TRAIN (Direct Read - Zero Disk Usage)
echo "🔥 Training $NAME (Reading from source)..."
python3 train.py \\
  --epochs 30 \\
  --data_dir "$DIRECT_DATA_PATH" \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --weights $WEIGHTS \\
  --loss $LOSS \\
  --im_size 512 \\
  --batch_size 4 \\
  --save_name $NAME

# 4. EVALUATE
echo "📊 Evaluating $NAME..."
python3 evaluate_official.py \\
  --data_dir "$DIRECT_DATA_PATH" \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --save_name $NAME\\
  --im_size 512

echo "✅ Done."
EOT

    # Submit with dependency on SETUP only (not previous training)
    # This allows them to run in PARALLEL once setup is done.
    sbatch --dependency=afterany:$SETUP_JOB_ID run_${NAME}.sh
}

echo "📋 Queueing Parallel Jobs..."

# Group A: U-Net
submit_parallel_job "unet_baseline_512" "unet" "resnet34" "imagenet" "bce"
submit_parallel_job "unet_scratch_512" "unet" "resnet34" "None" "bce"
submit_parallel_job "unet_dice_512" "unet" "resnet34" "imagenet" "dice"
submit_parallel_job "unet_deepsup_512" "deepsup" "resnet34" "imagenet" "bce"

# Group B: SegFormer
submit_parallel_job "segformer_b0_baseline_512" "segformer" "mit_b0" "imagenet" "bce"
submit_parallel_job "segformer_b2_capacity_512" "segformer" "mit_b2" "imagenet" "bce"
submit_parallel_job "segformer_b0_scratch_512" "segformer" "mit_b0" "None" "bce"
submit_parallel_job "segformer_b0_dice_512" "segformer" "mit_b0" "imagenet" "dice"

echo "----------------------------------------"
echo "🎉 All jobs submitted in PARALLEL mode."
echo "They will start as soon as 'SetupEnv' finishes."