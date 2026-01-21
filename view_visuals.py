import h5py
import matplotlib.pyplot as plt
import numpy as np

# Change this to match the file you downloaded
FILENAME = "visuals_unet_baseline.h5"

try:
    with h5py.File(FILENAME, 'r') as f:
        keys = list(f.keys())
        n_samples = len(keys)
        
        fig, axes = plt.subplots(n_samples, 3, figsize=(10, 3*n_samples))
        plt.subplots_adjust(wspace=0.1, hspace=0.2)
        
        for i, key in enumerate(keys):
            grp = f[key]
            img = grp['image'][:]
            gt = grp['gt'][:]
            pred = grp['pred'][:]
            
            # 1. Original Image
            ax = axes[i, 0] if n_samples > 1 else axes[0]
            ax.imshow(img)
            ax.set_title("Input Image")
            ax.axis('off')
            
            # 2. Ground Truth Mask
            ax = axes[i, 1] if n_samples > 1 else axes[1]
            ax.imshow(gt, cmap='gray')
            ax.set_title("Ground Truth")
            ax.axis('off')
            
            # 3. Prediction (Thresholded)
            ax = axes[i, 2] if n_samples > 1 else axes[2]
            # Show heat map or binary mask
            ax.imshow(pred > 0.5, cmap='gray') 
            ax.set_title("Prediction")
            ax.axis('off')
            
        plt.tight_layout()
        plt.show()
        
except FileNotFoundError:
    print(f"❌ File {FILENAME} not found. Did you download it?")