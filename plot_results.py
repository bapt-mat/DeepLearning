import h5py
import matplotlib.pyplot as plt
import os
import numpy as np

EXPERIMENTS = {
    # study 1: Architecture Comparison (U-Net vs SegFormer)
    "Architecture": {
        "files": [
            "results_unet_baseline.h5",
            "results_segformer_b0_baseline.h5",
            "results_segformer_b2_capacity.h5"
        ],
        "labels": ["U-Net (ResNet34)", "SegFormer B0", "SegFormer B2"],
        "title": "Ablation: Model Architecture"
    },
    
    # study 2: Loss Function (BCE vs Dice)
    "Loss Function": {
        "files": [
            "results_unet_baseline.h5",
            "results_unet_dice.h5",
            "results_segformer_b0_baseline.h5",
            "results_segformer_b0_dice.h5"
        ],
        "labels": ["U-Net (BCE)", "U-Net (Dice)", "SegFormer (BCE)", "SegFormer (Dice)"],
        "title": "Ablation: Loss Function Impact"
    },

    # study 3: Pre-training vs Scratch
    "Pre-training": {
        "files": [
            "results_unet_baseline.h5",
            "results_unet_scratch.h5",
            "results_segformer_b0_baseline.h5",
            "results_segformer_b0_scratch.h5"
        ],
        "labels": ["U-Net (ImageNet)", "U-Net (Scratch)", "SegFormer (ImageNet)", "SegFormer (Scratch)"],
        "title": "Ablation: Impact of Pre-training"
    },

    # study 4: Data Augmentation
    "Augmentation": {
        "files": [
            "results_unet_baseline.h5",
            "results_unet_aug.h5"
        ],
        "labels": ["Standard (Resize Only)", "Augmented (Flip/Rotate)"],
        "title": "Ablation: Data Augmentation"
    }
}

def load_history(filename):
    # loading history from H5 file
    if not os.path.exists(filename):
        print(f"Warning: {filename} not found")
        return None
    
    data = {}
    try:
        with h5py.File(filename, "r") as f:
            for k in f.keys():
                data[k] = np.array(f[k])
    except Exception as e:
        print(f"Error reading {filename}: {e}")
        return None
    return data

def get_key(history, candidates):
    # get key from history matching candidates
    for key in candidates:
        if key in history: return key
        for h_key in history.keys():
            if h_key.lower() == key.lower(): return h_key
    return None

def plot_ablation(experiment_name, config):
    # function to plot ablation study results
    print(f"Plotting {experiment_name}...")
    files = config["files"]
    labels = config["labels"]
    
    # Create 2 subplots: training loss and validation metric
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    has_data = False
    
    for f_name, label in zip(files, labels):
        history = load_history(f_name)
        if history is None: continue
        
        # find keys
        loss_key = get_key(history, ['loss', 'train_loss', 'Train_Loss', 'bce_loss'])
        val_key = get_key(history, ['val_dice', 'val_score', 'val_iou', 'val_loss'])
        
        if loss_key is None:
            print(f"Skipping {f_name}: No loss key found.")
            continue
            
        has_data = True
        epochs = range(1, len(history[loss_key]) + 1)
        
        # training loss plot
        ax1.plot(epochs, history[loss_key], label=label, linewidth=2)
        
        # validation plot
        if val_key:
            ax2.plot(epochs, history[val_key], label=f"{label}", linewidth=2, linestyle='--')
            ylabel = "Validation Score (Dice)" if 'dice' in val_key.lower() else "Validation Loss"
        else:
            ylabel = "Metric"

    if not has_data:
        plt.close()
        return

    ax1.set_title(f"{config['title']} - Training Loss")
    ax1.set_xlabel("Epochs")
    ax1.set_ylabel("Loss (Lower is better)")
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)

    ax2.set_title(f"{config['title']} - Validation Performance")
    ax2.set_xlabel("Epochs")
    ax2.set_ylabel(ylabel)
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    # Save
    safe_name = experiment_name.replace(" ", "_").lower()
    plt.savefig(f"plot_{safe_name}.png", dpi=300)
    print(f"Saved plot_{safe_name}.png")

if __name__ == "__main__":
    for name, config in EXPERIMENTS.items():
        plot_ablation(name, config)