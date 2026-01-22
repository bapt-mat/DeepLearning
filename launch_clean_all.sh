#!/bin/bash

# Variable to track the previous job ID
LAST_JOB_ID=""

# Function to create and submit a job with dependencies
submit_chained_job() {
    NAME=$1
    ARCH=$2
    ENCODER=$3
    WEIGHTS=$4
    LOSS=$5
    
    # 1. Create the Sbatch Script (Same "Space-Saver" version as before)
    cat <<EOT > run_${NAME}.sh
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/${NAME}_%j.log
#SBATCH --error=logs/${NAME}_err_%j.log
#SBATCH --job-name=$NAME

# -------------------------------------------------------
# SETUP & CLEANUP
# -------------------------------------------------------
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "\$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="\$SLURM_TMPDIR"; fi

LOCAL_DATA="\$TMPDIR/dataset_${NAME}"
VENV_PATH="\$TMPDIR/venv_${NAME}"

# Cleanup function (Runs on success or failure)
cleanup() {
    echo "🧹 CLEANUP: Deleting \$LOCAL_DATA and \$VENV_PATH"
    rm -rf \$LOCAL_DATA
    rm -rf \$VENV_PATH
}
trap cleanup EXIT

# -------------------------------------------------------
# EXECUTION
# -------------------------------------------------------
source /home_expes/tools/python/python3915_0_gpu/bin/activate

echo "🔧 Creating venv for $NAME..."
python3 -m venv \$VENV_PATH
PYBIN="\$VENV_PATH/bin/python3"
PIP="\$VENV_PATH/bin/pip"

echo "📦 Installing libraries..."
\$PIP install --no-cache-dir --upgrade pip
\$PIP install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm albumentations scikit-learn pandas numba --extra-index-url https://download.pytorch.org/whl/cu113

echo "🚀 Unpacking data..."
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
mkdir -p \$LOCAL_DATA
tar cf - -C \$SOURCE_DATA . | tar xf - -C \$LOCAL_DATA

echo "🔥 Training $NAME..."
\$PYBIN train.py \\
  --epochs 100 \\
  --data_dir \$LOCAL_DATA \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --weights $WEIGHTS \\
  --loss $LOSS \\
  --save_name $NAME

echo "📊 Evaluating $NAME..."
\$PYBIN evaluate_official.py \\
  --data_dir \$LOCAL_DATA \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --save_name $NAME

echo "✅ Done."
EOT

    # 2. SUBMIT THE JOB (With Dependency Logic)
    if [ -z "$LAST_JOB_ID" ]; then
        # First job: Submit normally
        JOB_ID=$(sbatch --parsable run_${NAME}.sh)
        echo "🚀 Started Chain with: $NAME (ID: $JOB_ID)"
    else
        # Subsequent jobs: Wait for the previous one to finish (afterany = success or fail)
        JOB_ID=$(sbatch --parsable --dependency=afterany:$LAST_JOB_ID run_${NAME}.sh)
        echo "🔗 Chained: $NAME (ID: $JOB_ID) -> waits for $LAST_JOB_ID"
    fi

    # Update LAST_JOB_ID so the next job waits for this one
    LAST_JOB_ID=$JOB_ID
}

# ==========================================
# 🧪 EXPERIMENT LIST (The Chain)
# ==========================================

# Group 1: U-Net
submit_chained_job "unet_baseline_100"       "unet"    "resnet34" "imagenet" "bce"
submit_chained_job "unet_scratch_100"        "unet"    "resnet34" "None"     "bce"
submit_chained_job "unet_dice_100"           "unet"    "resnet34" "imagenet" "dice"
submit_chained_job "unet_deepsup_100"        "deepsup" "resnet34" "imagenet" "bce"

# Group 2: SegFormer
submit_chained_job "segformer_b0_baseline_100" "segformer" "mit_b0" "imagenet" "bce"
submit_chained_job "segformer_b2_capacity_100" "segformer" "mit_b2" "imagenet" "bce"
submit_chained_job "segformer_b0_scratch_100"  "segformer" "mit_b0" "None"     "bce"
submit_chained_job "segformer_b0_dice_100"     "segformer" "mit_b0" "imagenet" "dice"

echo "---------------------------------------------------"
echo "🎉 All jobs submitted! They will run one by one."
echo "Use 'squeue -u <username>' to see the dependencies."