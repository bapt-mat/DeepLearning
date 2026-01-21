import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
# change this to match your filename
MODEL_NAME = "unet_baseline_long" 
HISTORY_FILE = f"history_{MODEL_NAME}.h5"
VISUALS_FILE = f"visuals_{MODEL_NAME}.h5"

def plot_history():
    """Plots Loss and Dice curves from the training history."""
    if not os.path.exists(HISTORY_FILE):
        print(f"⚠️  File not found: {HISTORY_FILE} (Skipping history plot)")
        return

    print(f"📊 Plotting training history for {MODEL_NAME}...")
    with h5py.File(HISTORY_FILE, 'r') as f:
        # Load data
        epochs = f['epochs'][:]
        loss = f['loss'][:]         # Training Loss
        val_loss = f['val_loss'][:] # Validation Loss
        val_dice = f['val_dice'][:] # Validation Dice Score

        # Create Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Loss
        ax1.plot(epochs, loss, label='Train Loss', color='gray', linestyle='--', alpha=0.6)
        ax1.plot(epochs, val_loss, label='Val Loss', color='red', linewidth=2)
        ax1.set_title("BCE Loss (Lower is Better)")
        ax1.set_xlabel("Epochs")
        ax1.set_ylabel("Loss")
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Dice Score
        ax2.plot(epochs, val_dice, label='Validation Dice', color='blue', linewidth=2)
        ax2.axhline(y=0.5, color='green', linestyle=':', label="0.5 Threshold")
        ax2.set_title("Dice Score (Higher is Better)")
        ax2.set_xlabel("Epochs")
        ax2.set_ylabel("Dice Coefficient")
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        plt.suptitle(f"Training Dynamics: {MODEL_NAME}", fontsize=16)
        plt.tight_layout()
        
        # Save
        save_path = f"plot_history_{MODEL_NAME}.png"
        plt.savefig(save_path, dpi=300)
        print(f"✅ Saved chart: {save_path}")
        plt.close()

def plot_visuals():
    """Plots a grid of Input vs Ground Truth vs Prediction."""
    if not os.path.exists(VISUALS_FILE):
        print(f"⚠️  File not found: {VISUALS_FILE} (Skipping visual plot)")
        return

    print(f"📸 Plotting visual predictions for {MODEL_NAME}...")
    with h5py.File(VISUALS_FILE, 'r') as f:
        keys = sorted(list(f.keys()))
        n_samples = len(keys)
        
        # Create Grid: Rows=Samples, Cols=3 (Img, GT, Pred)
        fig, axes = plt.subplots(n_samples, 3, figsize=(12, 4 * n_samples))
        plt.subplots_adjust(top=0.95, hspace=0.3)
        
        fig.suptitle(f"Qualitative Results: {MODEL_NAME}", fontsize=16)

        for i, key in enumerate(keys):
            grp = f[key]
            img = grp['image'][:]
            gt = grp['gt'][:]
            pred = grp['pred'][:]
            
            # --- Column 1: Input Image ---
            ax = axes[i, 0] if n_samples > 1 else axes[0]
            ax.imshow(img)
            if i == 0: ax.set_title("Input Image", fontweight='bold')
            ax.set_ylabel(key, fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            # --- Column 2: Ground Truth ---
            ax = axes[i, 1] if n_samples > 1 else axes[1]
            ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
            if i == 0: ax.set_title("Ground Truth", fontweight='bold')
            ax.axis('off')

            # --- Column 3: Prediction ---
            # We display it as a heatmap to see confidence, or threshold it
            ax = axes[i, 2] if n_samples > 1 else axes[2]
            
            # Show Binary Mask (> 0.5)
            pred_binary = (pred > 0.5).astype(float)
            ax.imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
            
            # Optional: Add confidence score text
            max_conf = np.max(pred)
            ax.text(10, 25, f"Max Conf: {max_conf:.2f}", color='yellow', fontsize=9, backgroundcolor='black')

            if i == 0: ax.set_title("Prediction (Binary)", fontweight='bold')
            ax.axis('off')

        save_path = f"plot_visuals_{MODEL_NAME}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Saved grid: {save_path}")
        plt.close()

if __name__ == "__main__":
    plot_history()
    print("-" * 30)
    plot_visuals() #ok