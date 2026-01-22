#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/pipeline_%j.log
#SBATCH --job-name=PipeEval

# 1. SETUP ENV
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="$SLURM_TMPDIR"; fi

source /home_expes/tools/python/python3915_0_gpu/bin/activate

# Reuse your existing environment (fastest)
if [ -d "venv_unet_baseline" ]; then
    source venv_unet_baseline/bin/activate
else
    # Fallback creation
    python3 -m venv $TMPDIR/venv
    source $TMPDIR/venv/bin/activate
    pip install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm scikit-learn pandas --extra-index-url https://download.pytorch.org/whl/cu113
fi

# 2. DATA TRANSFER
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
LOCAL_DATA="$TMPDIR/dataset_pipe"
mkdir -p $LOCAL_DATA
tar cf - -C $SOURCE_DATA . | tar xf - -C $LOCAL_DATA

# 3. RUN PIPELINE
echo "🔥 Running Two-Stage Pipeline Evaluation..."

# CLASSIFIER = segformer_b2_capacity
# SEGMENTER  = unet_baseline

python3 evaluate_pipeline.py \
    --data_dir $LOCAL_DATA \
    --cls_model "segformer_b2_capacity" \
    --cls_arch "segformer" \
    --cls_enc "mit_b2" \
    --seg_model "unet_baseline" \
    --seg_arch "unet" \
    --seg_enc "resnet34"

echo "✅ Pipeline evaluation finished."