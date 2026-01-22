#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=02:00:00
#SBATCH --output=logs/eval_all_%j.log
#SBATCH --job-name=EvalAll

# 1. SETUP ENV
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="$SLURM_TMPDIR"; fi

source /home_expes/tools/python/python3915_0_gpu/bin/activate

# Reuse existing environment or create new one
if [ -d "venv_unet_baseline" ]; then
    source venv_unet_baseline/bin/activate
else
    python3 -m venv $TMPDIR/venv
    source $TMPDIR/venv/bin/activate
    # IMPORTANT: Install scikit-learn for metrics
    pip install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm scikit-learn pandas --extra-index-url https://download.pytorch.org/whl/cu113
fi

# 2. DATA TRANSFER
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
LOCAL_DATA="$TMPDIR/dataset_eval"
mkdir -p $LOCAL_DATA
tar cf - -C $SOURCE_DATA . | tar xf - -C $LOCAL_DATA

# 3. LIST MODELS
MODELS=(
    "unet_baseline"
    "unet_dice"
    "unet_scratch"
    "unet_deepsup"
    "segformer_b0_baseline"
    "segformer_b0_dice"
    "segformer_b0_scratch"
    "segformer_b2_capacity"
)

# 4. EVALUATION LOOP
for MODEL in "${MODELS[@]}"; do
    if [ -f "${MODEL}.pth" ]; then
        echo "📊 Evaluating $MODEL..."
        
        # Configure Arch/Encoder
        ARCH="unet"
        ENCODER="resnet34"
        if [[ "$MODEL" == *"deepsup"* ]]; then ARCH="deepsup"; fi
        if [[ "$MODEL" == *"segformer"* ]]; then 
            ARCH="segformer"
            ENCODER="mit_b0"
            if [[ "$MODEL" == *"b2"* ]]; then ENCODER="mit_b2"; fi
        fi
        
        python3 evaluate_full_metrics.py \
            --data_dir $LOCAL_DATA \
            --save_name $MODEL \
            --arch $ARCH \
            --encoder $ENCODER
    else
        echo "⚠️  ${MODEL}.pth not found."
    fi
done

echo "✅ All evaluations complete."