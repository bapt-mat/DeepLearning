#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/thresh_study_%j.log
#SBATCH --job-name=ThreshStudy

export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128
if [ -z "$SLURM_TMPDIR" ]; then export TMPDIR="/tmp"; else export TMPDIR="$SLURM_TMPDIR"; fi

source /home_expes/tools/python/python3915_0_gpu/bin/activate

# Setup venv
if [ -d "venv_segformer_b2_capacity" ]; then
    source venv_segformer_b2_capacity/bin/activate
else
    python3 -m venv $TMPDIR/venv
    source $TMPDIR/venv/bin/activate
    pip install --no-cache-dir --upgrade "numpy<2" h5py opencv-python-headless torch==1.12.1+cu113 "segmentation-models-pytorch>=0.3.3" timm scikit-learn pandas matplotlib --extra-index-url https://download.pytorch.org/whl/cu113
fi

# Data Transfer
SOURCE_DATA="/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
LOCAL_DATA="$TMPDIR/dataset_thresh"
mkdir -p $LOCAL_DATA
tar cf - -C $SOURCE_DATA . | tar xf - -C $LOCAL_DATA

# RUN STUDY on your BEST MODEL (Segformer B2)
python3 study_threshold.py \
    --data_dir $LOCAL_DATA \
    --model_path "segformer_b2_capacity.pth" \
    --arch "segformer" \
    --encoder "mit_b2"