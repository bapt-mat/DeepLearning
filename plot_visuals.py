import h5py
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import glob

def plot_h5(file_path):
    #  plot visuals from h5 file
    save_name = os.path.basename(file_path).replace('visuals_', '').replace('.h5', '')
    print(f"Processing {file_path}...")
    
    try:
        with h5py.File(file_path, 'r') as f:
            images = f['images'][:]  
            masks = f['masks'][:] 
            preds = f['predictions'][:] 
            
            num_samples = len(images)
            if num_samples == 0:
                print("Empty file.")
                return

            limit = min(num_samples, 10)
            
            fig, axes = plt.subplots(3, 10, figsize=(25, 8))
            
            plt.subplots_adjust(wspace=0.1, hspace=0.3)

            for i in range(10):
                if i >= limit:
                    axes[0, i].axis('off')
                    axes[1, i].axis('off')
                    axes[2, i].axis('off')
                    continue

                # row 1: Input Image
                axes[0, i].imshow(images[i])
                if i == 0: axes[0, i].set_ylabel("Input", fontsize=14, fontweight='bold')
                axes[0, i].set_title(f"Sample {i+1}", fontsize=9)
                axes[0, i].set_xticks([])
                axes[0, i].set_yticks([])

                # row 2: Ground Truth
                axes[1, i].imshow(masks[i], cmap='gray')
                if i == 0: axes[1, i].set_ylabel("Ground Truth", fontsize=14, fontweight='bold')
                axes[1, i].set_xticks([])
                axes[1, i].set_yticks([])

                # row 3: Prediction & Classification
                max_prob = preds[i].max()
                is_forged = max_prob > 0.5
                
                label_text = "FORGED" if is_forged else "AUTHENTIC"
                label_color = "red" if is_forged else "green"

                axes[2, i].imshow(preds[i], cmap='jet', vmin=0, vmax=1)
                if i == 0: axes[2, i].set_ylabel("Prediction", fontsize=14, fontweight='bold')
                
                # classif text
                axes[2, i].set_xlabel(f"{label_text}\n({max_prob:.2f})", 
                                      fontsize=10, 
                                      color=label_color, 
                                      fontweight='bold')
                axes[2, i].set_xticks([])
                axes[2, i].set_yticks([])

            # Save
            out_name = f"final_plot_{save_name}.png"
            plt.savefig(out_name, dpi=150, bbox_inches='tight')
            plt.close()
            print(f"Saved horizontal grid to {out_name}")

    except Exception as e:
        print(f"Error reading {file_path}: {e}")

if __name__ == "__main__":
    files = sorted(glob.glob("visuals_*.h5"))
    if not files:
        print("No 'visuals_*.h5' files found")
    else:
        for f in files:
            plot_h5(f)