import h5py
import matplotlib.pyplot as plt
import os
import numpy as np
import argparse

# --- HELPER FUNCTIONS ---

def load_history(filename):
    if not os.path.exists(filename):
        print(f"Warning: '{filename}' not found. Skipping.")
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
    for key in candidates:
        if key in history: return key
        for h_key in history.keys():
            if h_key.lower() == key.lower(): return h_key
    return None


def run_plot(files, labels, title, output_file, mode):
    print(f"Initializing Plot: {title}")
    print(f"Mode: {mode}")
    
    if mode == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        ax_loss, ax_val = axes[0], axes[1]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        ax_loss = ax if mode == 'loss' else None
        ax_val = ax if mode == 'val' else None

    has_data = False
    
    if not labels or len(labels) != len(files):
        if labels: print("Label count mismatch. Using filenames as labels.")
        labels = [os.path.basename(f).replace('.h5', '').replace('results_', '') for f in files]

    # Iterate files
    for f_name, label in zip(files, labels):
        history = load_history(f_name)
        if history is None: continue
        
        loss_key = get_key(history, ['loss', 'train_loss', 'bce_loss'])
        val_key = get_key(history, ['val_dice', 'val_score', 'val_iou', 'val_loss', 'dice'])
        
        epochs = None
        if loss_key: epochs = range(1, len(history[loss_key]) + 1)
        elif val_key: epochs = range(1, len(history[val_key]) + 1)
        
        if not epochs:
            print(f"Skipping {f_name}: No recognizable data found.")
            continue

        has_data = True
        
        if ax_loss and loss_key:
            ax_loss.plot(epochs, history[loss_key], label=label, linewidth=2)
        elif ax_loss:
            print(f"{f_name}: 'loss' key missing.")

        if ax_val and val_key:
            ax_val.plot(epochs, history[val_key], label=label, linewidth=2, linestyle='--')
        elif ax_val:
            print(f"{f_name}: Validation key missing.")

    if not has_data:
        print("No valid data plotted, ciao")
        plt.close()
        return

    if ax_loss:
        ax_loss.set_title(f"{title} - Training Loss")
        ax_loss.set_xlabel("Epochs")
        ax_loss.set_ylabel("Loss")
        ax_loss.legend()
        ax_loss.grid(True, linestyle='--', alpha=0.6)

    if ax_val:
        y_lbl = "Score (Dice)" 
        ax_val.set_title(f"{title} - Validation")
        ax_val.set_xlabel("Epochs")
        ax_val.set_ylabel(y_lbl)
        ax_val.legend()
        ax_val.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    
    if not output_file.endswith('.png'): output_file += '.png'
    plt.savefig(output_file, dpi=300)
    print(f"Saved plot to: {output_file}")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--files', nargs='+', required=True, help="List of .h5 files to plot")
    parser.add_argument('--labels', nargs='+', help="Legend labels (must match number of files)")
    parser.add_argument('--title', type=str, default="Model Comparison", help="Main title of the plot")
    parser.add_argument('--out', type=str, default="plot.png", help="Output filename (e.g. comparison.png)")
    parser.add_argument('--mode', type=str, choices=['both', 'loss', 'val'], default='both', 
                        help="What to plot: 'loss' (train only), 'val' (validation only), or 'both' (side-by-side)")

    args = parser.parse_args()
    
    run_plot(args.files, args.labels, args.title, args.out, args.mode)