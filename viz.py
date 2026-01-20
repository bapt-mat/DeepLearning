import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from model import SimpleUNet
from dataset import ForgeryDataset

# --- SETTINGS ---
MODEL_PATH = "model_epoch_3.pth"  # Update this to your best model
DATA_PATH = "./data" # Update local path
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_SAMPLES = 10
PIXEL_THRESHOLD = 0.5
AREA_THRESHOLD = 10 # Number of pixels required to call it "Forged"

def visualize_results():
    print(f"Loading model from {MODEL_PATH}...")
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Loading dataset from {DATA_PATH}...")
    val_ds = ForgeryDataset(DATA_PATH, phase='val')

    # 1. Filter for Forged Images only (so we have interesting masks to compare)
    forged_indices = []
    for i in range(len(val_ds)):
        # Access internal list directly for speed to find label 1
        if val_ds.dataset[i][1] == 1: 
            forged_indices.append(i)

    if not forged_indices:
        print("No forged images found in validation set!")
        return

    # 2. Select Random Samples
    num_display = min(NUM_SAMPLES, len(forged_indices))
    selected_indices = random.sample(forged_indices, num_display)
    
    print(f"Plotting {num_display} images...")

    # 3. Create Grid Plot (3 Rows x N Columns)
    # Increased height slightly to make room for the text below
    fig, axes = plt.subplots(3, num_display, figsize=(20, 7))
    
    # Row Labels
    rows = ['Original Image', 'Ground Truth Mask', 'Predicted Mask']
    for ax, row in zip(axes[:,0], rows):
        ax.set_ylabel(row, rotation=90, size='large', weight='bold')

    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            img_tensor, mask_tensor = val_ds[idx]
            
            # --- PREDICTION ---
            input_batch = img_tensor.unsqueeze(0).to(DEVICE)
            output = model(input_batch)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            
            # --- CLASSIFICATION LOGIC ---
            pred_bin = (pred_prob > PIXEL_THRESHOLD).astype(np.uint8)
            white_pixels = np.sum(pred_bin)
            
            if white_pixels > AREA_THRESHOLD:
                pred_class = "FORGED"
                color = "red" # Highlight Forged in red
            else:
                pred_class = "AUTHENTIC"
                color = "green" # Highlight Authentic in green (error in this case since we know inputs are forged)

            # Prepare images for display
            orig_img = img_tensor.permute(1, 2, 0).numpy()
            gt_mask = mask_tensor.squeeze().numpy()
            
            # --- ROW 1: Original ---
            axes[0, i].imshow(orig_img)
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            axes[0, i].set_title(f"ID: {idx}")

            # --- ROW 2: Ground Truth ---
            axes[1, i].imshow(gt_mask, cmap='gray', vmin=0, vmax=1)
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])

            # --- ROW 3: Prediction & Classification ---
            axes[2, i].imshow(pred_prob, cmap='gray', vmin=0, vmax=1)
            axes[2, i].set_xticks([])
            axes[2, i].set_yticks([])
            
            # Add the Classification Label BELOW the image
            axes[2, i].set_xlabel(f"{pred_class}\n({white_pixels} px)", fontsize=12, fontweight='bold', color=color)

    plt.tight_layout()
    plt.savefig("results_grid_classified.png")
    print("✅ Saved to results_grid_classified.png")
    plt.show()

if __name__ == "__main__":
    visualize_results()