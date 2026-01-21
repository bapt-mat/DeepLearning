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
    
    # We will loop until we find 5 good ones, or check the first 2000 images
    for i in range(min(len(val_ds), 2000)):
        
        # Quick check: If the dataset has a label list, skip authentic (0) immediately
        # (This assumes your dataset class has a .images attribute list)
        if hasattr(val_ds, 'images'):
             _, label, _ = val_ds.images[i]
             if label == 0: continue

        # Load the actual tensors
        img_tensor, mask_tensor = val_ds[i]
        
        # --- CRITICAL CHECK ---
        # Only keep this image if the mask has white pixels (sum > 0)
        if mask_tensor.sum() > 10:  # Threshold > 10 pixels to avoid tiny noise
            indices_to_save.append(i)
            print(f"   ✅ Found valid sample at index {i} (Mask pixels: {mask_tensor.sum().item()})")
            
            if len(indices_to_save) >= 5:
                break
        else:
            # If label was 1 but mask is empty, we print a warning (debugging)
            if hasattr(val_ds, 'images') and label == 1:
                print(f"   ⚠️  Index {i} is labeled Forged but mask is empty! Skipping.")

    if len(indices_to_save) == 0:
        print("❌ Error: Could not find ANY samples with valid masks. Check your dataset path/loading logic.")
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