import os
import glob
from plot_custom import run_plot

output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)


baseline_files = [
    "results_unet_baseline.h5",
    "results_segformer_b0_baseline.h5"
]
baseline_labels = ["U-Net Pre-trained", "SegFormer B0 Pre-trained"]
run_plot(baseline_files, baseline_labels, "Baseline Performance", os.path.join(output_dir, "baseline_comparison.png"), mode='both')

# --- U-Net Ablations 
# Pre-training
unet_pretrain_files = ["results_unet_baseline.h5", "results_unet_scratch.h5"]
unet_pretrain_labels = ["Pre-trained (Baseline)", "Scratch (No Pre-training)"]
run_plot(unet_pretrain_files, unet_pretrain_labels, "U-Net Ablation: Pre-training", os.path.join(output_dir, "unet_pretraining.png"), mode='both')

# Loss function
unet_loss_files = ["results_unet_baseline.h5", "results_unet_dice.h5"]
unet_loss_labels = ["Dice Loss (Baseline)", "BCE Loss"]
run_plot(unet_loss_files, unet_loss_labels, "U-Net Ablation: Loss Function", os.path.join(output_dir, "unet_loss.png"), mode='both')

# Deep Supervision
unet_deepsup_files = ["results_unet_baseline.h5", "results_unet_deepsup.h5"]
unet_deepsup_labels = ["Baseline", "Deep Supervision"]
run_plot(unet_deepsup_files, unet_deepsup_labels, "U-Net Ablation: Deep Supervision", os.path.join(output_dir, "unet_deepsup.png"), mode='both')

# --- SegFormer Ablations 
# Architecture capacity
segformer_arch_files = ["results_segformer_b0_baseline.h5", "results_segformer_b2_capacity.h5"]
segformer_arch_labels = ["MiT-B0 (Baseline)", "MiT-B2"]
run_plot(segformer_arch_files, segformer_arch_labels, "SegFormer Ablation: Architecture Capacity", os.path.join(output_dir, "segformer_architecture.png"), mode='both')

# Pre-training
segformer_pretrain_files = ["results_segformer_b0_baseline.h5", "results_segformer_b0_scratch.h5"]
segformer_pretrain_labels = ["Pre-trained (Baseline)", "Scratch"]
run_plot(segformer_pretrain_files, segformer_pretrain_labels, "SegFormer Ablation: Pre-training", os.path.join(output_dir, "segformer_pretraining.png"), mode='both')

# Data augmentation
segformer_aug_files = ["results_segformer_b0_baseline.h5", "results_segformer_b2_aug.h5"]
segformer_aug_labels = ["No Augmentation (Baseline)", "With Augmentation"]
run_plot(segformer_aug_files, segformer_aug_labels, "SegFormer Ablation: Data Augmentation", os.path.join(output_dir, "segformer_augmentation.png"), mode='both')

# Input resolution
segformer_res_files = ["results_segformer_b0_baseline.h5", "results_segformer_b2_512.h5"]
segformer_res_labels = ["256x256 (Baseline)", "512x512"]
run_plot(segformer_res_files, segformer_res_labels, "SegFormer Ablation: Input Resolution", os.path.join(output_dir, "segformer_resolution.png"), mode='both')

# --- Training Duration Ablations 
training_duration_files = [
    "results_unet_baseline.h5", "results_unet_baseline_100.h5",
    "results_segformer_b0_baseline.h5", "results_segformer_b0_baseline_100.h5",
    "results_segformer_b2_capacity.h5", "results_segformer_b2_capacity_100.h5"
]
training_duration_labels = [
    "U-Net Baseline 30", "U-Net Baseline 100",
    "SegFormer B0 30", "SegFormer B0 100",
    "SegFormer B2 30", "SegFormer B2 100"
]
run_plot(training_duration_files, training_duration_labels, "Training Duration Ablation", os.path.join(output_dir, "training_duration.png"), mode='both')

print("All plots generated in folder:", output_dir)
