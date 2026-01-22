import h5py
import matplotlib.pyplot as plt
import numpy as np
import glob
import os

def plot_model_visuals(filename):
    # Extract model name from filename (e.g., "visuals_unet_baseline.h5" -> "unet_baseline")
    model_name = filename.replace("visuals_", "").replace(".h5", "")
    print(f"🎨 Plotting results for: {model_name}")
    
    try:
        with h5py.File(filename, 'r') as f:
            keys = sorted(list(f.keys()))
            n_samples = len(keys)
            
            if n_samples == 0:
                print(f"⚠️  No samples found in {filename}. Skipping.")
                return

            # --- SETUP FIGURE ---
            # 3 Rows (Input, GT, Pred) x N Columns (Samples)
            # Width adjusts dynamically based on how many samples we have
            fig, axes = plt.subplots(3, n_samples, figsize=(n_samples * 3, 11))
            plt.subplots_adjust(wspace=0.1, hspace=0.3)
            
            fig.suptitle(f"Qualitative Analysis: {model_name}", fontsize=20, y=0.98, fontweight='bold')
            
            # Handle case where n_samples is 1 (axes becomes 1D array)
            if n_samples == 1:
                axes = axes[:, np.newaxis]

            for i, key in enumerate(keys):
                grp = f[key]
                img = grp['image'][:]
                gt = grp['gt'][:]
                pred = grp['pred'][:]
                
                # --- Row 1: Input Image ---
                ax = axes[0, i]
                ax.imshow(img)
                if i == 0: ax.set_ylabel("Input Image", fontsize=14, fontweight='bold')
                ax.set_title(f"Sample {i+1}", fontsize=12)
                ax.set_xticks([])
                ax.set_yticks([])

                # --- Row 2: Ground Truth ---
                ax = axes[1, i]
                ax.imshow(gt, cmap='gray', vmin=0, vmax=1)
                if i == 0: ax.set_ylabel("Ground Truth", fontsize=14, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])

                # --- Row 3: Prediction ---
                ax = axes[2, i]
                
                # We threshold at 0.5 for a clean binary mask view
                pred_binary = (pred > 0.5).astype(float)
                ax.imshow(pred_binary, cmap='gray', vmin=0, vmax=1)
                
                if i == 0: ax.set_ylabel("Prediction", fontsize=14, fontweight='bold')
                ax.set_xticks([])
                ax.set_yticks([])
                
                # --- CLASSIFICATION LABEL ---
                # Logic: If the model is >50% confident on ANY pixel, we call it "Forged"
                max_conf = np.max(pred)
                is_forged = max_conf > 0.5
                
                label_text = "FORGED" if is_forged else "AUTHENTIC"
                label_color = "#d62728" if is_forged else "#2ca02c" # Red vs Green
                
                # Add text below the image
                ax.set_xlabel(f"Pred: {label_text}\nMax Conf: {max_conf:.2f}", 
                              fontsize=11, fontweight='bold', color=label_color)

            # Save the plot
            save_path = f"qualitative_{model_name}.png"
            plt.savefig(save_path, bbox_inches='tight', dpi=150)
            print(f"✅ Saved chart: {save_path}")
            plt.close()

    except Exception as e:
        print(f"❌ Error reading {filename}: {e}")

if __name__ == "__main__":
    # 1. Find all files matching "visuals_*.h5"
    files = glob.glob("visuals_*.h5")
    
    if not files:
        print("❌ No 'visuals_*.h5' files found in this folder.")
        print("   -> Did you download them from the cluster?")
    else:
        print(f"🔎 Found {len(files)} files. Starting batch plotting...")
        print("-" * 40)
        for f in sorted(files):
            plot_model_visuals(f)
        print("-" * 40)
        print("🎉 All plots generated!")