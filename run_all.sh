#!/bin/bash

# Function to create and submit a job
submit_experiment() {
    NAME=$1
    ARCH=$2
    ENCODER=$3
    WEIGHTS=$4
    LOSS=$5
    
    echo "🚀 Submitting Job: $NAME ($ARCH | $ENCODER | $WEIGHTS | $LOSS)"

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

# 1. SETUP ENV & PROXY
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "\$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="\$SLURM_TMPDIR"; fi

# Load Base Python
source /home_expes/tools/python/python3915_0_gpu/bin/activate

# Create VENV in TMPDIR
if [ ! -d "\$TMPDIR/venv" ]; then 
    echo "🔧 Creating Virtual Environment..."
    python3 -m venv \$TMPDIR/venv
fi

# Define paths to force usage of the VENV
PYBIN="\$TMPDIR/venv/bin/python3"
PIP="\$TMPDIR/venv/bin/pip"

# 2. INSTALL DEPENDENCIES (Using the VENV pip)
echo "📦 Installing libraries..."
\$PIP install --no-cache-dir --upgrade pip
\$PIP install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm --extra-index-url https://download.pytorch.org/whl/cu113

# Debug: Check version
\$PYBIN -c "import segmentation_models_pytorch; print(f'SMP Version: {segmentation_models_pytorch.__version__}')"

# 3. DATA TRANSFER (Unique Folder per Job)
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
LOCAL_DATA="\$TMPDIR/dataset_${NAME}" 

echo "🚀 Unpacking data to \$LOCAL_DATA..."
mkdir -p \$LOCAL_DATA
tar cf - -C \$SOURCE_DATA . | tar xf - -C \$LOCAL_DATA

# 4. TRAIN (100 Epochs) - Using \$PYBIN explicitly
echo "🔥 Training $NAME..."
\$PYBIN train.py \\
  --epochs 100 \\
  --data_dir \$LOCAL_DATA \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --weights $WEIGHTS \\
  --loss $LOSS \\
  --save_name $NAME

# 5. EVALUATE - Using \$PYBIN explicitly
echo "📊 Evaluating $NAME..."
\$PYBIN evaluate_official.py \\
  --data_dir \$LOCAL_DATA \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --save_name $NAME

# 6. CLEANUP
rm -rf \$LOCAL_DATA

echo "✅ Done."
EOT

    # Submit the generated script
    sbatch run_${NAME}.sh
}

# ==========================================
# 🧪 GROUP A: U-NET FAMILY
# ==========================================
submit_experiment "unet_baseline" "unet" "resnet34" "imagenet" "bce"
submit_experiment "unet_scratch" "unet" "resnet34" "None" "bce"
submit_experiment "unet_dice" "unet" "resnet34" "imagenet" "dice"
submit_experiment "unet_deepsup" "deepsup" "resnet34" "imagenet" "bce"

# ==========================================
# 🧪 GROUP B: SEGFORMER FAMILY
# ==========================================
submit_experiment "segformer_b0_baseline" "segformer" "mit_b0" "imagenet" "bce"
submit_experiment "segformer_b2_capacity" "segformer" "mit_b2" "imagenet" "bce"
submit_experiment "segformer_b0_scratch" "segformer" "mit_b0" "None" "bce"
submit_experiment "segformer_b0_dice" "segformer" "mit_b0" "imagenet" "dice"

echo "----------------------------------------"
echo "🎉 All 8 experiments relaunched using EXPLICIT python paths."