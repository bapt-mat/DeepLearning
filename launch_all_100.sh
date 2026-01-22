#!/bin/bash

# Function to create and submit a job
submit_experiment() {
    NAME=$1
    ARCH=$2
    ENCODER=$3
    WEIGHTS=$4
    LOSS=$5
    
    echo "🚀 Submitting Job: $NAME"

    # Create a temporary SLURM script
    cat <<EOT > run_${NAME}.sh
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/${NAME}_%j.log
#SBATCH --error=logs/${NAME}_err_%j.log
#SBATCH --job-name=$NAME

# 1. SETUP PROXY & ENV VARIABLES
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "\$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="\$SLURM_TMPDIR"; fi

# 2. CREATE UNIQUE VIRTUAL ENVIRONMENT (Fixed: Unique Name)
# We use the job name to create a unique venv for THIS specific job
VENV_PATH="\$TMPDIR/venv_${NAME}"
PYBIN="\$VENV_PATH/bin/python3"
PIP="\$VENV_PATH/bin/pip"

# Load base python to create the venv
source /home_expes/tools/python/python3915_0_gpu/bin/activate

echo "🔧 Creating isolated environment at \$VENV_PATH..."
python3 -m venv \$VENV_PATH

# 3. INSTALL DEPENDENCIES (Into the unique venv)
echo "📦 Installing libraries..."
\$PIP install --no-cache-dir --upgrade pip
\$PIP install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm --extra-index-url https://download.pytorch.org/whl/cu113

# 4. DATA TRANSFER (Unique Folder)
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
LOCAL_DATA="\$TMPDIR/dataset_${NAME}" 

echo "🚀 Unpacking data to \$LOCAL_DATA..."
mkdir -p \$LOCAL_DATA
tar cf - -C \$SOURCE_DATA . | tar xf - -C \$LOCAL_DATA

# 5. TRAIN (100 Epochs) - Using explicit python path
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

# 7. CLEANUP
rm -rf \$LOCAL_DATA
rm -rf \$VENV_PATH

echo "✅ Done."
EOT

    # Submit
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
echo "🎉 All 8 experiments relaunched with suffix _100 and FULL ISOLATION."