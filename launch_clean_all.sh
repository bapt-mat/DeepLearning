#!/bin/bash

# Path to the Shared Venv we created in Step 1
SHARED_VENV="$HOME/DeepForg/venv_shared"

# Function to submit a job
submit_job() {
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

# 1. SETUP
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "\$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="\$SLURM_TMPDIR"; fi

# 2. ACTIVATE SHARED VENV
if [ -f "$SHARED_VENV/bin/activate" ]; then
    source $SHARED_VENV/bin/activate
else
    echo "❌ Error: Shared Venv not found at $SHARED_VENV"
    echo "   Did you run Step 1 (setup_env.sh)?"
    exit 1
fi

# 3. PREPARE DATA (Local Scratch)
LOCAL_DATA="\$TMPDIR/dataset_${NAME}"
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

# Cleanup Trap (Deletes data immediately after job ends to save space)
cleanup() {
    rm -rf \$LOCAL_DATA
}
trap cleanup EXIT

echo "🚀 Unpacking data..."
mkdir -p \$LOCAL_DATA
tar cf - -C \$SOURCE_DATA . | tar xf - -C \$LOCAL_DATA

# 4. TRAIN
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

echo "✅ Done."
EOT

    # Submit
    sbatch run_${NAME}.sh
}

# ==========================================
# 🧪 SUBMIT ALL JOBS (PARALLEL)
# ==========================================

# Group 1: U-Net
submit_job "unet_baseline_100"       "unet"    "resnet34" "imagenet" "bce"
submit_job "unet_scratch_100"        "unet"    "resnet34" "None"     "bce"
submit_job "unet_dice_100"           "unet"    "resnet34" "imagenet" "dice"
submit_job "unet_deepsup_100"        "deepsup" "resnet34" "imagenet" "bce"

# Group 2: SegFormer
submit_job "segformer_b0_baseline_100" "segformer" "mit_b0" "imagenet" "bce"
submit_job "segformer_b2_capacity_100" "segformer" "mit_b2" "imagenet" "bce"
submit_job "segformer_b0_scratch_100"  "segformer" "mit_b0" "None"     "bce"
submit_job "segformer_b0_dice_100"     "segformer" "mit_b0" "imagenet" "dice"

echo "----------------------------------------"
echo "🎉 All 8 jobs submitted using SHARED VENV."