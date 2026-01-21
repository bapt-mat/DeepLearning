import h5py
import matplotlib.pyplot as plt
import glob
import os

# 1. Find all visual files
files = sorted(glob.glob("visuals_*.h5"))

if not files:
    print("❌ No 'visuals_*.h5' files found! Make sure you downloaded them to this folder.")
    exit()

print(f"found {len(files)} model files: {files}")

# 2. Get the list of sample keys (sample_0, sample_1, etc.) from the first file
with h5py.File(files[0], 'r') as f:
    sample_keys = list(f.keys())

# 3. Iterate through each sample image (we saved 5 samples)
for sample_key in sample_keys:
    print(f"📊 Generating comparison for {sample_key}...")
    
    n_models = len(files)
    fig, axes = plt.subplots(n_models, 3, figsize=(10, 3 * n_models))
    plt.subplots_adjust(top=0.95, bottom=0.05, hspace=0.4)
    
    fig.suptitle(f"Visual Comparison: {sample_key}", fontsize=16)

    for row, filename in enumerate(files):
        model_name = filename.replace("visuals_", "").replace(".h5", "")
        
        try:
            with h5py.File(filename, 'r') as f:
                if sample_key not in f: continue
                
                grp = f[sample_key]
                img = grp['image'][:]
                gt = grp['gt'][:]
                pred = grp['pred'][:]
                
                # Column 1: Original Image (Only need to label it once)
                ax = axes[row, 0]
                ax.imshow(img)
                if row == 0: ax.set_title("Input Image", fontsize=10)
                ax.set_ylabel(model_name, rotation=90, size='large', weight='bold') # Model Name on the left
                ax.set_xticks([])
                ax.set_yticks([])
                
                # Column 2: Ground Truth
                ax = axes[row, 1]
                ax.imshow(gt, cmap='gray')
                if row == 0: ax.set_title("Ground Truth", fontsize=10)
                ax.axis('off')
                
                # Column 3: Prediction
                ax = axes[row, 2]
                ax.imshow(pred > 0.5, cmap='gray') # Binary threshold
                if row == 0: ax.set_title("Prediction", fontsize=10)
                ax.axis('off')

        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # Save the huge grid
    save_name = f"comparison_{sample_key}.png"
    plt.savefig(save_name, bbox_inches='tight', dpi=150)
    print(f"✅ Saved {save_name}")
    plt.close()