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

# 1. SETUP PROXY
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "\$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="\$SLURM_TMPDIR"; fi

# 2. ACTIVATE SHARED ENVIRONMENT
# We point to the master environment created in Step 1
VENV_PATH="\$HOME/venv_master_100"
if [ ! -d "\$VENV_PATH" ]; then
    echo "❌ Error: Master environment not found at \$VENV_PATH. Please run setup_env.sh first!"
    exit 1
fi
source \$VENV_PATH/bin/activate
echo "✅ Activated shared environment: \$VENV_PATH"

# 3. DATA TRANSFER (Unique Temp Folder)
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
LOCAL_DATA="\$TMPDIR/dataset_${NAME}" 

echo "🚀 Unpacking data to \$LOCAL_DATA..."
mkdir -p \$LOCAL_DATA
# We use 'tar' but ignore errors slightly in case of weird permissions, though space should be fine now.
tar cf - -C \$SOURCE_DATA . | tar xf - -C \$LOCAL_DATA

# 4. TRAIN (100 Epochs)
echo "🔥 Training $NAME..."
python3 train.py \\
  --epochs 100 \\
  --data_dir \$LOCAL_DATA \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --weights $WEIGHTS \\
  --loss $LOSS \\
  --save_name $NAME

# 5. EVALUATE
echo "📊 Evaluating $NAME..."
python3 evaluate_official.py \\
  --data_dir \$LOCAL_DATA \\
  --arch $ARCH \\
  --encoder $ENCODER \\
  --save_name $NAME

# 6. CLEANUP
rm -rf \$LOCAL_DATA
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
echo "🎉 All 8 experiments submitted using SHARED ENV to save disk space."