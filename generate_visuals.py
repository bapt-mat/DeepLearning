import torch
import h5py
import argparse
import numpy as np
from dataset import ForgeryDataset
from model import FlexibleModel

def save_visuals():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_name', type=str, required=True)
    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet34')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Load Model
    print(f"🔄 Loading model: {args.save_name}...")
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None).to(device)
    try:
        model.load_state_dict(torch.load(f"{args.save_name}.pth", map_location=device))
    except FileNotFoundError:
        print(f"❌ Error: Weights file '{args.save_name}.pth' not found.")
        return
    
    model.eval()

    # 2. Load Dataset
    val_ds = ForgeryDataset(args.data_dir, phase='val')
    print(f"📂 Dataset loaded with {len(val_ds)} images.")
    print("🔎 Scanning for samples with visible Ground Truth masks...")

    # 3. Find Images with ACTUAL WHITE PIXELS in the mask
    indices_to_save = []
    
    # Increased safety limit since we want more images
    max_checks = min(len(val_ds), 4000)
    
    for i in range(max_checks):
        try:
            # Load the actual tensors via the standard method (Safe)
            img_tensor, mask_tensor = val_ds[i]
            
            # --- FILTER: Check for white pixels ---
            if mask_tensor.sum() > 10: 
                indices_to_save.append(i)
                print(f"   ✅ Found valid sample at index {i} (Mask pixels: {mask_tensor.sum().item()})")
                
                # --- CHANGED HERE: Stop after finding 10 ---
                if len(indices_to_save) >= 10: 
                    break
                    
        except Exception as e:
            # print(f"   ⚠️ Error loading index {i}: {e}") # Optional: Comment out to reduce noise
            continue

    if len(indices_to_save) == 0:
        print("❌ Error: Could not find ANY samples with valid masks.")
        return

    # 4. Save Predictions
    with h5py.File(f"visuals_{args.save_name}.h5", "w") as f:
        print(f"📸 Saving {len(indices_to_save)} samples to visuals_{args.save_name}.h5...")
        
        for i, idx in enumerate(indices_to_save):
            img_tensor, mask_tensor = val_ds[idx]
            
            # Predict
            with torch.no_grad():
                input_t = img_tensor.unsqueeze(0).to(device)
                out = model(input_t)
                if isinstance(out, list): out = out[0]
                pred_prob = torch.sigmoid(out).squeeze().cpu().numpy()
            
            # Prepare data
            img_np = img_tensor.permute(1, 2, 0).numpy()
            mask_np = mask_tensor.squeeze().numpy()
            
            grp = f.create_group(f"sample_{i}")
            grp.create_dataset("image", data=img_np)
            grp.create_dataset("gt", data=mask_np)
            grp.create_dataset("pred", data=pred_prob)
            
    print("✅ Done!")

if __name__ == "__main__":
    save_visuals()