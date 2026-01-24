#!/bin/bash

# ==========================================
# ⚙️ CONFIGURATION
# ==========================================
# 1. Path to the SHARED Environment (Must match what you built)
SHARED_VENV="$HOME/DeepForg/venv_shared"

# 2. Path to the ORIGINAL DATASET (We read directly from here)
# This points to the folder on the cluster, so we don't need to copy it.
DIRECT_DATA_PATH="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

# ==========================================
# 🛠️ STEP 1: VERIFY ENVIRONMENT
# ==========================================
if [ ! -d "$SHARED_VENV" ]; then
    echo "❌ Error: Shared Venv not found at $SHARED_VENV"
    echo "   You must run the setup script (00_setup_env.sh) first to build the libraries."
    exit 1
fi

# ==========================================
# 🔗 STEP 2: SUBMIT EXPERIMENTS
# ==========================================
# We still use chaining (Sequential) to be safe with GPU availability,
# but now we don't worry about disk space at all.

LAST_JOB_ID=""

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

# 2. ACTIVATE SHARED VENV
source $SHARED_VENV/bin/activate

# 3. TRAIN (Direct Read)
# We point --data_dir DIRECTLY to the source. No copying.
echo "🔥 Training $NAME reading from Source..."

python3 train.py \\
  --epochs 100 \\
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
    if [ -z "$LAST_JOB_ID" ]; then
        JOB_ID=$(sbatch --parsable run_${NAME}.sh)
        echo "🚀 Started: $NAME (ID: $JOB_ID)"
    else
        JOB_ID=$(sbatch --parsable --dependency=afterany:$LAST_JOB_ID run_${NAME}.sh)
        echo "🔗 Chained: $NAME (ID: $JOB_ID)"
    fi
    LAST_JOB_ID=$JOB_ID
}

echo "📋 Queueing 'Direct Read' jobs..."

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
echo "🎉 All jobs submitted."
echo "Running directly from source (No Copying)."