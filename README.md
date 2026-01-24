# DeepLearning - Image Forgery Detection

A deep learning project for detecting forged regions in images using semantic segmentation models. This is a M2 (Master's level) project implementing state-of-the-art architectures for image manipulation detection.

## 📋 Overview

This project implements and compares various deep learning architectures for image forgery detection:
- **UNet** with ResNet encoders (ResNet34, ResNet50)
- **SegFormer** (B0, B2 variants)
- **Custom Deep Supervision ResUNet**

The models are trained to segment tampered regions in images, classifying each pixel as authentic or forged.

## 🚀 Features

- Multiple architecture options (UNet, SegFormer, DeepSupervision)
- Pre-trained encoder support (ImageNet weights)
- Multiple loss functions (BCE, Dice Loss)
- Data augmentation support (optional, using Albumentations)
- Comprehensive evaluation metrics (Dice score, Kaggle-format RLE)
- SLURM cluster job submission scripts
- Results visualization and analysis tools
- Threshold optimization studies

## 🛠️ Installation

### Requirements

- Python 3.9+
- PyTorch
- CUDA-capable GPU (recommended)

### Setup

```bash
# Clone the repository
git clone https://github.com/bapt-mat/DeepLearning.git
cd DeepLearning

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision
pip install segmentation-models-pytorch
pip install opencv-python numpy h5py pandas scipy tqdm

# Optional: For data augmentation
pip install albumentations
```

## 📂 Project Structure

```
.
├── train.py                      # Main training script
├── model.py                      # Model architectures (UNet, SegFormer, Deep Supervision ResUNet)
├── dataset.py                    # Dataset loading and augmentation
├── evaluate_official.py          # Kaggle-format evaluation
├── evaluate_full_metrics.py      # Comprehensive metrics evaluation
├── evaluate_pipeline.py          # Full pipeline evaluation
├── generate_visuals.py           # Generate visualization results
├── plot_results.py              # Plot training/validation curves
├── plot_visuals.py              # Visualize predictions
├── study_threshold.py           # Threshold optimization
├── kaggle_metric.py             # Kaggle competition metrics
├── launch_*.sh                  # SLURM job submission scripts
└── results_*.h5                 # Saved experiment results
```

## 🎓 Usage

### Training a Model

Basic training with default settings (UNet + ResNet34 + ImageNet weights):

```bash
python train.py --data_dir /path/to/dataset --epochs 20 --save_name unet_baseline
```

Advanced training options:

```bash
# SegFormer B0 with Dice loss
python train.py \
  --data_dir /path/to/dataset \
  --arch segformer \
  --encoder b0 \
  --weights imagenet \
  --loss dice \
  --epochs 20 \
  --save_name segformer_b0_dice

# UNet from scratch (no pretrained weights)
python train.py \
  --data_dir /path/to/dataset \
  --arch unet \
  --encoder resnet34 \
  --weights None \
  --loss bce \
  --epochs 20 \
  --save_name unet_scratch

# UNet with data augmentation
USE_AUGMENTATION=True python train.py \
  --data_dir /path/to/dataset \
  --arch unet \
  --encoder resnet34 \
  --save_name unet_aug
```

### Training Arguments

- `--data_dir`: Path to dataset directory (required)
- `--epochs`: Number of training epochs (default: 20)
- `--arch`: Architecture choice: `unet`, `segformer`, `deepsup` (default: unet)
- `--encoder`: Encoder backbone: `resnet34`, `resnet50`, `b0`, `b2` (default: resnet34)
- `--weights`: Pre-trained weights: `imagenet`, `None` (default: imagenet)
- `--loss`: Loss function: `bce`, `dice` (default: bce)
- `--save_name`: Name for saving model and results (default: model)

### Data Augmentation

Enable augmentation by setting the environment variable:

```bash
export USE_AUGMENTATION=True
python train.py --data_dir /path/to/dataset --save_name model_aug
```

Augmentation includes:
- Horizontal and vertical flips
- Random 90° rotations
- Random shift, scale, and rotation
- ImageNet normalization

### Evaluation

Evaluate a trained model using Kaggle-format metrics:

```bash
python evaluate_official.py \
  --data_dir /path/to/dataset \
  --arch unet \
  --encoder resnet34 \
  --save_name unet_baseline
```

Get comprehensive metrics:

```bash
python evaluate_full_metrics.py --save_name unet_baseline
```

### Visualization

Generate prediction visualizations:

```bash
python generate_visuals.py \
  --data_dir /path/to/dataset \
  --arch unet \
  --encoder resnet34 \
  --save_name unet_baseline
```

Plot training curves:

```bash
python plot_results.py
```

Plot visual predictions:

```bash
python plot_visuals.py
```

### Threshold Optimization

Study optimal prediction threshold:

```bash
python study_threshold.py \
  --data_dir /path/to/dataset \
  --arch segformer \
  --encoder b2 \
  --save_name segformer_b2_capacity
```

## 🖥️ SLURM Cluster Usage

For HPC cluster environments with SLURM:

```bash
# Submit all experiments
./launch_all.sh

# Submit single experiment
sbatch launch_unet_aug.sh

# Evaluate all models
sbatch launch_eval_all.sh

# Generate visualizations
sbatch launch_all_visuals.sh
```

## 📊 Dataset Format

The expected dataset structure:

```
data_dir/
├── train_images/
│   ├── authentic/
│   │   ├── image1.png
│   │   └── ...
│   └── forged/
│       ├── image1.png
│       └── ...
└── train_masks/
    ├── image1.npy
    └── ...
```

- Images should be PNG format
- Masks should be NumPy arrays (.npy) indicating forged regions
- The dataset is automatically split 80/20 for train/validation

## 🏗️ Model Architectures

### UNet
- Encoder: ResNet34/ResNet50 (with ImageNet pretraining)
- Decoder: Symmetric upsampling with skip connections
- Output: Single-channel segmentation mask

### SegFormer
- Hierarchical transformer encoder (B0, B2 variants)
- Lightweight MLP decoder
- Multi-scale feature fusion

### Deep Supervision ResUNet
- Custom residual blocks in encoder/decoder
- Multiple auxiliary outputs for deep supervision
- Enhanced gradient flow during training

## 📈 Results

Training results are saved in HDF5 format containing:
- Training loss history
- Validation metrics (Dice score)
- Per-epoch performance

Visualization results include:
- Original images
- Ground truth masks
- Predicted masks
- Overlay comparisons

## 🔧 Configuration

### Environment Variables

- `USE_AUGMENTATION`: Enable/disable data augmentation (default: False)
- `HTTP_PROXY`, `HTTPS_PROXY`: Proxy settings for cluster environments

### Output Files

- `{save_name}.pth`: Trained model weights
- `results_{save_name}.h5`: Training metrics and results
- `visuals_{save_name}.h5`: Visualization predictions
- `logs/`: SLURM job logs

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{deeplearning-forgery,
  author = {Baptiste Mathon},
  title = {Image Forgery Detection using Deep Learning},
  year = {2024},
  publisher = {GitHub},
  url = {https://github.com/bapt-mat/DeepLearning}
}
```

## 📄 License

This project is available for academic and research purposes.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 👤 Author

Baptiste Mathon - M2 Deep Learning Project
