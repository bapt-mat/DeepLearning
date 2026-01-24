#!/bin/bash

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
SHARED_VENV="$HOME/DeepForg/venv_shared"
# Points directly to the cluster storage (No disk copying = No space error)
DIRECT_DATA_PATH="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
SLURM_TMP_VAR="\$SLURM_TMPDIR" 

# ==========================================
# 🧹 STEP 0: FORCE CLEANUP
# ==========================================
# We delete the venv to ensure we aren't using a broken one.
echo "🧹 Deleting old environment to ensure a clean start..."
rm -rf "$SHARED_VENV"

# ==========================================
# 🛠️ STEP 1: SUBMIT SETUP JOB
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

# 1. Load Base Python (Sets LD_LIBRARY_PATH)
source /home_expes/tools/python/python3915_0_gpu/bin/activate

# 2. Create Venv
python3 -m venv $SHARED_VENV

# 3. Install Libraries
source $SHARED_VENV/bin/activate
pip install --no-cache-dir --upgrade pip
# CRITICAL: --no-cache-dir prevents filling up your home directory
pip install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm albumentations scikit-learn pandas numba --extra-index-url https://download.pytorch.org/whl/cu113

echo "✅ Environment Ready."
EOT

# Submit Setup Job
SETUP_JOB_ID=$(sbatch --parsable setup_env.sh)
echo "🚀 Submitted Setup Job (ID: $SETUP_JOB_ID)"

# ==========================================
# 🔗 STEP 2: SUBMIT CHAINED EXPERIMENTS
# ==========================================
LAST_JOB_ID=$SETUP_JOB_ID

submit_direct_job() {
    NAME=$1
    ARCH=$2
    ENCODER=$3
    WEIGHTS=$4
    LOSS=$5
    
    # Create the job script
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

# 2. ACTIVATE ENV (THE FIX)
# First: Load the base python to get the shared libraries (libpython3.9.so)
source /home_expes/tools/python/python3915_0_gpu/bin/activate

# Second: Activate our specific environment
source $SHARED_VENV/bin/activate

# 3. TRAIN (Direct Read - Zero Disk Usage)
echo "🔥 Training $NAME..."
python3 train.py \\
  --epochs 30 \\
  --data_dir "$DIRECT_DATA_PATH" \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --weights $WEIGHTS \\
  --loss $LOSS \\
  --save_name $NAME

# 4. EVALUATE
echo "📊 Evaluating $NAME..."
python3 evaluate_official.py \\
  --data_dir "$DIRECT_DATA_PATH" \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --save_name $NAME

echo "✅ Done."
EOT

    # Submit with dependency
    JOB_ID=$(sbatch --parsable --dependency=afterany:$LAST_JOB_ID run_${NAME}.sh)
    echo "🔗 Chained: $NAME (ID: $JOB_ID) -> waits for $LAST_JOB_ID"
    LAST_JOB_ID=$JOB_ID
}

echo "📋 Queueing jobs..."

# --- U-Net Experiments ---
submit_direct_job "unet_baseline"       "unet"    "resnet34" "imagenet" "bce"
submit_direct_job "unet_scratch"        "unet"    "resnet34" "None"     "bce"
submit_direct_job "unet_dice"           "unet"    "resnet34" "imagenet" "dice"
submit_direct_job "unet_deepsup"        "deepsup" "resnet34" "imagenet" "bce"

# --- SegFormer Experiments ---
submit_direct_job "segformer_b0_baseline" "segformer" "mit_b0" "imagenet" "bce"
submit_direct_job "segformer_b2_capacity" "segformer" "mit_b2" "imagenet" "bce"
submit_direct_job "segformer_b0_scratch"  "segformer" "mit_b0" "None"     "bce"
submit_direct_job "segformer_b0_dice"     "segformer" "mit_b0" "imagenet" "dice"

echo "---------------------------------------------------"
echo "🎉 Fixed script submitted."
echo "1. Old Venv deleted."
echo "2. Setup Job creating new Venv."
echo "3. Training jobs queued (Fixed library path)."