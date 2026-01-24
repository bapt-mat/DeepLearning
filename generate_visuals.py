import torch
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
import cv2
from dataset import ForgeryDataset
from model import FlexibleModel

def save_visual(img_tensor, mask_tensor, pred_tensor, save_path, title):
    """Helper to save a side-by-side plot"""
    # 1. Un-normalize image for display
    img = img_tensor.permute(1, 2, 0).cpu().numpy()
    
    # 2. Prepare masks
    gt = mask_tensor.squeeze().cpu().numpy()
    pred = torch.sigmoid(pred_tensor).squeeze().cpu().numpy()
    
    # 3. Plot
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
    ax[0].imshow(img)
    ax[0].set_title("Input Image")
    ax[0].axis('off')
    
    ax[1].imshow(gt, cmap='gray')
    ax[1].set_title("Ground Truth")
    ax[1].axis('off')
    
    ax[2].imshow(pred, cmap='jet', vmin=0, vmax=1)
    ax[2].set_title(f"Prediction (Prob Map)\n{title}")
    ax[2].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def run_visuals():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_name', type=str, required=True) # e.g. unet_baseline
    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet34')
    args = parser.parse_args()

    # Create output folder
    os.makedirs("visuals", exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"📸 Generating visuals for: {args.save_name}")

    # 1. Load Model
    # Handle 'deepsup' or other variants
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None, n_classes=1).to(device)
    
    weights_path = f"{args.save_name}.pth"
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights file '{weights_path}' not found.")
        return

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except Exception as e:
        print(f"⚠️ Warning: Direct load failed, trying loose load... {e}")
        # Sometimes models saved with DataParallel need 'module.' stripped, or strict=False
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        
    model.eval()

    # 2. Load Dataset
    val_ds = ForgeryDataset(args.data_dir, phase='val')
    
    # --- THE FIX: INTELLIGENT SCANNING ---
    # Instead of iterating 0..N, we look at the dataset metadata directly
    # val_ds.dataset is a list of (path, label, mask_path)
    
    # Find indices of FORGED images (label == 1)
    forged_indices = [i for i, x in enumerate(val_ds.dataset) if x[1] == 1]
    
    if len(forged_indices) == 0:
        print("❌ Error: No forged images found in validation set.")
        return

    print(f"✅ Found {len(forged_indices)} forged samples in validation set.")
    
    # Select 3 random forged images
    np.random.seed(42)
    selected_indices = np.random.choice(forged_indices, 3, replace=False)

    # 3. Generate Predictions
    with torch.no_grad():
        for i, idx in enumerate(selected_indices):
            img, mask = val_ds[idx]
            input_t = img.unsqueeze(0).to(device)
            
            # Predict
            out = model(input_t)
            if isinstance(out, list): out = out[0] # Handle deep supervision
            
            # Calculate Dice for title
            pred_bin = (torch.sigmoid(out) > 0.5).float()
            intersection = (pred_bin * mask.to(device)).sum()
            dice = (2. * intersection) / (pred_bin.sum() + mask.to(device).sum() + 1e-6)
            
            # Save
            out_file = f"visuals/{args.save_name}_sample_{i}.png"
            save_visual(img, mask, out, out_file, title=f"Dice: {dice.item():.4f}")
            print(f"   👉 Saved {out_file}")

if __name__ == "__main__":
    run_visuals()