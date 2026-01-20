import torch
import numpy as np
import matplotlib.pyplot as plt
import random
from model import SimpleUNet
from dataset import ForgeryDataset

# --- SETTINGS ---
MODEL_PATH = "model_epoch_3.pth"  # Your downloaded model
DATA_PATH = "./data" # Your local path
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load Model
model = SimpleUNet().to(DEVICE)
# Map location allows loading a GPU model on a CPU (Mac)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# 2. Load Validation Set
val_ds = ForgeryDataset(DATA_PATH, phase='val')

# 3. Find Forged Indices
# We iterate to find images that actually have a mask (forgeries)
forged_indices = []
for i in range(len(val_ds)):
    _, label, _ = val_ds.dataset[i]  # Access internal list to be fast
    if label == 1:
        forged_indices.append(i)

if not forged_indices:
    print("No forged images found!")
    exit()

# 4. Select 10 Random Forgeries
num_samples = min(10, len(forged_indices))
selected_indices = random.sample(forged_indices, num_samples)

# 5. Plotting
plt.figure(figsize=(15, 5 * num_samples))

for i, idx in enumerate(selected_indices):
    img_tensor, mask_tensor = val_ds[idx]
    
    # Predict
    with torch.no_grad():
        input_batch = img_tensor.unsqueeze(0).to(DEVICE)
        output = model(input_batch)
        pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
    
    # Prepare Images for Display
    # Original: (C, H, W) -> (H, W, C)
    orig = img_tensor.permute(1, 2, 0).numpy()
    # Mask: (1, H, W) -> (H, W)
    gt = mask_tensor.squeeze().numpy()
    # Pred: Already (H, W)
    
    # --- Row i: Original ---
    plt.subplot(num_samples, 3, i * 3 + 1)
    plt.imshow(orig)
    plt.title(f"Original #{idx}")
    plt.axis('off')
    
    # --- Row i: Ground Truth ---
    plt.subplot(num_samples, 3, i * 3 + 2)
    plt.imshow(gt, cmap='gray', vmin=0, vmax=1)
    plt.title("Ground Truth Mask")
    plt.axis('off')

    # --- Row i: Prediction ---
    plt.subplot(num_samples, 3, i * 3 + 3)
    plt.imshow(pred_prob, cmap='gray', vmin=0, vmax=1)
    plt.title(f"Prediction (Max: {pred_prob.max():.2f})")
    plt.axis('off')

plt.tight_layout()
plt.savefig("visualization_results.png")
print("✅ Saved plot to visualization_results.png")
plt.show()