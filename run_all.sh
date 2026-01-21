#!/bin/bash

# Function to create and submit a job
submit_experiment() {
    NAME=$1
    ARCH=$2
    ENCODER=$3
    WEIGHTS=$4
    LOSS=$5
    
    echo "🚀 Submitting Job: $NAME ($ARCH | $ENCODER | $WEIGHTS | $LOSS)"

    # Create a temporary SLURM script for this specific experiment
    cat <<EOT > run_${NAME}.sh
#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=20:00:00
#SBATCH --output=logs/${NAME}_%j.log
#SBATCH --error=logs/${NAME}_err_%j.log
#SBATCH --job-name=$NAME

# 1. SETUP ENV
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "\$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="\$SLURM_TMPDIR"; fi

source /home_expes/tools/python/python3915_0_gpu/bin/activate
if [ ! -d "\$TMPDIR/venv" ]; then python3 -m venv \$TMPDIR/venv; fi
source \$TMPDIR/venv/bin/activate

# 2. INSTALL DEPENDENCIES (FIXED: Added upgrade + timm for SegFormer)
# We force upgrade segmentation-models-pytorch to get the 'Segformer' class
pip install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm --extra-index-url https://download.pytorch.org/whl/cu113

# 3. DATA TRANSFER (FIXED: Unique Folder per Job)
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
# We append the job name to the path to ensure isolation
LOCAL_DATA="\$TMPDIR/dataset_${NAME}" 

echo "🚀 Unpacking data to \$LOCAL_DATA..."
mkdir -p \$LOCAL_DATA
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

# 6. CLEANUP (Save space on /tmp)
rm -rf \$LOCAL_DATA

echo "✅ Done."
EOT

    # Submit the generated script
    sbatch run_${NAME}.sh
}

# ==========================================
# 🧪 GROUP A: U-NET FAMILY (4 Jobs)
# ==========================================
submit_experiment "unet_baseline" "unet" "resnet34" "imagenet" "bce"
submit_experiment "unet_scratch" "unet" "resnet34" "None" "bce"
submit_experiment "unet_dice" "unet" "resnet34" "imagenet" "dice"
submit_experiment "unet_deepsup" "deepsup" "resnet34" "imagenet" "bce"

# ==========================================
# 🧪 GROUP B: SEGFORMER FAMILY (4 Jobs)
# ==========================================
submit_experiment "segformer_b0_baseline" "segformer" "mit_b0" "imagenet" "bce"
submit_experiment "segformer_b2_capacity" "segformer" "mit_b2" "imagenet" "bce"
submit_experiment "segformer_b0_scratch" "segformer" "mit_b0" "None" "bce"
submit_experiment "segformer_b0_dice" "segformer" "mit_b0" "imagenet" "dice"

echo "----------------------------------------"
echo "🎉 All 8 experiments have been relaunched (Fixed Dependencies + Unique Paths)!"