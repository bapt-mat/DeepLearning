import h5py
import matplotlib.pyplot as plt
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_NAME = "unet_baseline_long" 
HISTORY_FILE = f"results_{MODEL_NAME}.h5"
VISUALS_FILE = f"visuals_{MODEL_NAME}.h5"

def plot_history():
    """Plots Loss and Dice curves from the training history."""
    if not os.path.exists(HISTORY_FILE):
        print(f"⚠️  File not found: {HISTORY_FILE} (Skipping history plot)")
        return

    print(f"📊 Plotting training history for {MODEL_NAME}...")
    
    try:
        with h5py.File(HISTORY_FILE, 'r') as f:
            epochs = f['epochs'][:]
            train_loss = f['train_loss'][:] 
            val_dice = f['val_dice'][:] 

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
            
            # Plot 1: Training Loss
            ax1.plot(epochs, train_loss, label='Train Loss', color='blue', linewidth=2)
            ax1.set_title("Training Loss (BCE)")
            ax1.set_xlabel("Epochs")
            ax1.set_ylabel("Loss")
            ax1.grid(True, alpha=0.3)
            ax1.legend()

            # Plot 2: Dice Score
            ax2.plot(epochs, val_dice, label='Validation Dice', color='green', linewidth=2)
            ax2.axhline(y=0.5, color='gray', linestyle=':', label="0.5 Threshold")
            
            # Mark Best Epoch
            best_idx = np.argmax(val_dice)
            ax2.scatter(epochs[best_idx], val_dice[best_idx], color='gold', s=100, edgecolors='black', zorder=5)
            ax2.text(epochs[best_idx], val_dice[best_idx] - 0.05, f"Best: {val_dice[best_idx]:.3f}", ha='center')

            ax2.set_title("Validation Dice Score")
            ax2.set_xlabel("Epochs")
            ax2.set_ylabel("Dice")
            ax2.set_ylim(0, 1.0)
            ax2.grid(True, alpha=0.3)
            ax2.legend()

            plt.suptitle(f"Training Dynamics: {MODEL_NAME}", fontsize=16)
            plt.tight_layout()
            plt.savefig(f"plot_history_{MODEL_NAME}.png", dpi=300)
            print(f"✅ Saved chart: plot_history_{MODEL_NAME}.png")
            plt.close()
            
    except Exception as e:
        print(f"❌ Error plotting history: {e}")

def plot_visuals():
    """Plots a 3-Row Grid (Horizontal Layout)."""
    if not os.path.exists(VISUALS_FILE):
        print(f"⚠️  File not found: {VISUALS_FILE} (Skipping visual plot)")
        return

    print(f"📸 Plotting visual predictions for {MODEL_NAME}...")
    with h5py.File(VISUALS_FILE, 'r') as f:
        keys = sorted(list(f.keys()))
        n_samples = len(keys)
        
        if n_samples == 0:
            print("⚠️  No samples found in visuals file.")
            return

        # --- HORIZONTAL LAYOUT SETUP ---
        # 3 Rows (Input, GT, Pred) x N Columns (Samples)
        # We adjust figure width based on number of samples
        fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3, 10))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        fig.suptitle(f"Qualitative Results: {MODEL_NAME}", fontsize=16)

        # Handle 1-sample case (axes would be 1D)
        if n_samples == 1:
            axes = axes[:, np.newaxis]

        for i, key in enumerate(keys):
            grp = f[key]
            img = grp['image'][:]
            gt = grp['gt'][:]
            pred = grp['pred'][:]
            
            # Row 1: Input Image
            ax = axes[0, i]
            ax.imshow(img)
            if i == 0: ax.set_ylabel("Input Image", fontsize=12, fontweight='bold')
            ax.set_title(f"Sample {i+1}", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])

            # Row 2: Ground Truth
            ax = axes[1, i]
            ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
            if i == 0: ax.set_ylabel("Ground Truth", fontsize=12, fontweight='bold')
            ax.set_xticks([])
            ax.set_yticks([])

            # Row 3: Prediction
            ax = axes[2, i]
            pred_binary = (pred > 0.5).astype(float)
            ax.imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
            if i == 0: ax.set_ylabel("Prediction", fontsize=12, fontweight='bold')
            ax.set_yticks([])
            
            # --- CLASSIFICATION LOGIC ---
            # If max probability > 0.5, we classify it as Forged
            max_conf = np.max(pred)
            is_forged = max_conf > 0.5
            
            label_text = f"Pred: FORGED\n(Conf: {max_conf:.2f})" if is_forged else f"Pred: AUTHENTIC\n(Conf: {max_conf:.2f})"
            label_color = 'red' if is_forged else 'green'
            
            ax.set_xlabel(label_text, fontsize=10, fontweight='bold', color=label_color)

        save_path = f"plot_visuals_{MODEL_NAME}.png"
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        print(f"✅ Saved grid: {save_path}")
        plt.close()

if __name__ == "__main__":
    plot_history()
    print("-" * 30)
    plot_visuals()