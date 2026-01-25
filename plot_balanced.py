import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def plot_h5(file_path):
    save_name = os.path.basename(file_path).replace('balanced_visuals_', '').replace('.h5', '')
    print(f"📂 Processing {file_path}...")
    
    try:
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]
            masks = f['masks'][:]
            preds = f['predictions'][:]
            
            num_samples = len(images)
            limit = min(num_samples, 10)
            
            # Setup Plot: 3 Rows x 10 Columns
            fig, axes = plt.subplots(3, 10, figsize=(25, 8))
            plt.subplots_adjust(wspace=0.1, hspace=0.3)

            for i in range(10):
                if i >= limit:
                    for r in range(3): axes[r, i].axis('off')
                    continue

                # --- ROW 1: Input Image ---
                axes[0, i].imshow(images[i])
                
                # Dynamic Header
                if i < 5:
                    header = f"Sample {i+1}\n(GT: Authentic)"
                    color = 'green'
                else:
                    header = f"Sample {i+1}\n(GT: Forged)"
                    color = 'blue'
                
                axes[0, i].set_title(header, fontsize=9, color=color, fontweight='bold')
                
                if i == 0: axes[0, i].set_ylabel("Input", fontsize=14, fontweight='bold')
                axes[0, i].set_xticks([]); axes[0, i].set_yticks([])

                # --- ROW 2: Ground Truth ---
                axes[1, i].imshow(masks[i], cmap='gray')
                if i == 0: axes[1, i].set_ylabel("Ground Truth", fontsize=14, fontweight='bold')
                axes[1, i].set_xticks([]); axes[1, i].set_yticks([])

                # --- ROW 3: Prediction & Classification ---
                max_prob = preds[i].max()
                is_forged = max_prob > 0.5
                
                label_text = "PRED: FORGED" if is_forged else "PRED: AUTHENTIC"
                label_color = "red" if is_forged else "green"
                
                # Check for errors (False Positive / False Negative)
                # First 5 are Auth (GT=0), Last 5 are Forged (GT=1)
                is_gt_forged = (i >= 5)
                
                if is_gt_forged and not is_forged:
                    label_text += "\n(MISS)"
                    label_color = "purple"
                elif not is_gt_forged and is_forged:
                    label_text += "\n(FALSE ALARM)"
                    label_color = "orange"

                axes[2, i].imshow(preds[i], cmap='jet', vmin=0, vmax=1)
                if i == 0: axes[2, i].set_ylabel("Prediction", fontsize=14, fontweight='bold')
                
                axes[2, i].set_xlabel(f"{label_text}\nMax: {max_prob:.2f}", 
                                      fontsize=9, 
                                      color=label_color, 
                                      fontweight='bold')
                axes[2, i].set_xticks([]); axes[2, i].set_yticks([])

            # Save
            out_name = f"balanced_plot_{save_name}.png"
            plt.savefig(out_name, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"   ✅ Saved balanced grid to {out_name}")

    except Exception as e:
        print(f"   ❌ Error reading {file_path}: {e}")

if __name__ == "__main__":
    files = sorted(glob.glob("balanced_visuals_*.h5"))
    if not files:
        print("No 'balanced_visuals_*.h5' files found!")
    else:
        for f in files:
            plot_h5(f)