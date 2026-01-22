#!/bin/bash

# Function to submit a job with AUTO-CLEANUP
submit_experiment() {
    NAME=$1
    ARCH=$2
    ENCODER=$3
    WEIGHTS=$4
    LOSS=$5
    
    echo "🚀 Submitting Job: $NAME"

    cat <<EOT > run_${NAME}.sh
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/${NAME}_%j.log
#SBATCH --error=logs/${NAME}_err_%j.log
#SBATCH --job-name=$NAME

# 1. SETUP ENV & CLEANUP TRAP
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "\$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="\$SLURM_TMPDIR"; fi

# Define paths
LOCAL_DATA="\$TMPDIR/dataset_${NAME}"
VENV_PATH="\$TMPDIR/venv_${NAME}"

# SAFETY: If the job crashes or finishes, FORCE delete the data/venv
cleanup() {
    echo "🧹 TRAILING CLEANUP: Deleting \$LOCAL_DATA and \$VENV_PATH"
    rm -rf \$LOCAL_DATA
    rm -rf \$VENV_PATH
}
trap cleanup EXIT

# 2. CREATE VENV
# Load base python
source /home_expes/tools/python/python3915_0_gpu/bin/activate

echo "🔧 Creating isolated environment..."
python3 -m venv \$VENV_PATH
PYBIN="\$VENV_PATH/bin/python3"
PIP="\$VENV_PATH/bin/pip"

# 3. INSTALL DEPENDENCIES (Reduced size)
echo "📦 Installing libraries..."
\$PIP install --no-cache-dir --upgrade pip
\$PIP install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm albumentations scikit-learn pandas numba --extra-index-url https://download.pytorch.org/whl/cu113

# 4. DATA TRANSFER
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
echo "🚀 Unpacking data..."
mkdir -p \$LOCAL_DATA
tar cf - -C \$SOURCE_DATA . | tar xf - -C \$LOCAL_DATA

# 5. TRAIN (100 Epochs)
echo "🔥 Training $NAME..."
\$PYBIN train.py \\
  --epochs 100 \\
  --data_dir \$LOCAL_DATA \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --weights $WEIGHTS \\
  --loss $LOSS \\
  --save_name $NAME

# 6. EVALUATE
echo "📊 Evaluating $NAME..."
\$PYBIN evaluate_official.py \\
  --data_dir \$LOCAL_DATA \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --save_name $NAME

echo "✅ Done."
# Trap will handle cleanup automatically now
EOT

    sbatch run_${NAME}.sh
}

# ==========================================
# 🧪 GROUP A: U-NET FAMILY (100 Epochs)
# ==========================================
submit_experiment "unet_baseline_100" "unet" "resnet34" "imagenet" "bce"
submit_experiment "unet_scratch_100" "unet" "resnet34" "None" "bce"
submit_experiment "unet_dice_100" "unet" "resnet34" "imagenet" "dice"
submit_experiment "unet_deepsup_100" "deepsup" "resnet34" "imagenet" "bce"

# ==========================================
# 🧪 GROUP B: SEGFORMER FAMILY (100 Epochs)
# ==========================================
submit_experiment "segformer_b0_baseline_100" "segformer" "mit_b0" "imagenet" "bce"
submit_experiment "segformer_b2_capacity_100" "segformer" "mit_b2" "imagenet" "bce"
submit_experiment "segformer_b0_scratch_100" "segformer" "mit_b0" "None" "bce"
submit_experiment "segformer_b0_dice_100" "segformer" "mit_b0" "imagenet" "dice"

echo "----------------------------------------"
echo "🎉 Experiments relaunched with AUTO-CLEANUP."