#!/bin/bash
#SBATCH --partition=GPU
#SBATCH --gres=gpu:1
#SBATCH --mem=12G
#SBATCH --time=00:20:00
#SBATCH --output=logs/viz_512_%j.log
#SBATCH --job-name=Viz512

# 1. SETUP ENV
export HTTP_PROXY=http://cache.univ-st-etienne.fr:3128
export HTTPS_PROXY=http://cache.univ-st-etienne.fr:3128

SHARED_VENV="$HOME/DeepForg/venv_shared"

if [ -d "$SHARED_VENV" ]; then
    echo "✅ Activating Shared Venv..."
    source /home_expes/tools/python/python3915_0_gpu/bin/activate
    source $SHARED_VENV/bin/activate
else
    echo "❌ Error: Shared Venv not found."
    exit 1
fi

# 2. RUN VISUALIZATION
python3 generate_viz_512.py

echo "✅ Done."