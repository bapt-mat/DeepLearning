import torch
import numpy as np
import argparse
import os
import h5py
from dataset import ForgeryDataset
from model import FlexibleModel

def run_visuals():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_name', type=str, required=True)
    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--im_size', type=int, default=256, help="Model input resolution")
    
    # NEW ARGUMENTS
    parser.add_argument('--mode', type=str, choices=['find', 'use'], default='use', 
                        help="'find' = search for TP images and save indices. 'use' = load indices from file.")
    parser.add_argument('--indices_file', type=str, default='fixed_indices.npy', 
                        help="Path to save or load the indices.")
    
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💾 Visuals Mode: {args.mode} | Model: {args.save_name} ({args.im_size}px)")

    # 1. Load Model
    # Note: For DeepSup, we only care about the final output for visuals
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None, n_classes=1).to(device)
    
    weights_path = f"{args.save_name}.pth"
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights file '{weights_path}' not found.")
        return

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except:
        print(f"⚠️  Loading with strict=False")
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        
    model.eval()

    # 2. Load Dataset
    # We use the requested resolution for inference
    val_ds = ForgeryDataset(args.data_dir, phase='val', im_size=(args.im_size, args.im_size))
    
    selected_indices = []

    # --- LOGIC BRANCH A: FIND INDICES (Run this with B2_512) ---
    if args.mode == 'find':
        print("🔍 Scanning for 10 Correctly Detected Forgeries (True Positives)...")
        
        # Get all forged indices first to avoid scanning authentic ones
        # We access the internal list from dataset to be fast
        # (path, label, mask_path)
        all_indices = range(len(val_ds))
        
        found_count = 0
        with torch.no_grad():
            for i in all_indices:
                img, mask = val_ds[i]
                
                # Skip Authentic images immediately
                if mask.max() == 0: continue
                
                # Check Model Prediction
                input_t = img.unsqueeze(0).to(device)
                out = model(input_t)
                if isinstance(out, list): out = out[0]
                prob_map = torch.sigmoid(out).squeeze().cpu().numpy()
                
                # Criteria: Must have a strong detection (max prob > 0.8)
                if prob_map.max() > 0.8:
                    selected_indices.append(i)
                    found_count += 1
                    print(f"   ✅ Found TP at index {i} (Max Prob: {prob_map.max():.4f})")
                
                if found_count >= 10:
                    break
        
        # Save them
        if len(selected_indices) > 0:
            np.save(args.indices_file, np.array(selected_indices))
            print(f"💾 Saved {len(selected_indices)} indices to {args.indices_file}")
        else:
            print("❌ Could not find enough true positives!")
            return

    # --- LOGIC BRANCH B: USE INDICES (Run this with others) ---
    else:
        if not os.path.exists(args.indices_file):
            print(f"❌ Indices file {args.indices_file} not found! Run 'find' mode first.")
            return
        print(f"📂 Loading indices from {args.indices_file}...")
        selected_indices = np.load(args.indices_file)

    # 3. Generate Visuals for the Selected Indices
    store_imgs = []
    store_masks = []
    store_preds = []

    print(f"⚡ Processing {len(selected_indices)} samples...")
    with torch.no_grad():
        for idx in selected_indices:
            # Important: int(idx) is needed because numpy types can break pytorch datasets
            img, mask = val_ds[int(idx)] 
            
            input_t = img.unsqueeze(0).to(device)
            out = model(input_t)
            if isinstance(out, list): out = out[0]
            
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()
            
            # Store (Convert images back to 0-255 for storage efficiency)
            img_numpy = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask_numpy = mask.squeeze().numpy().astype(np.uint8)
            
            store_imgs.append(img_numpy)
            store_masks.append(mask_numpy)
            store_preds.append(prob_map)

    # 4. Save H5
    out_file = f"visuals_{args.save_name}.h5"
    with h5py.File(out_file, 'w') as f:
        f.create_dataset("images", data=np.array(store_imgs))
        f.create_dataset("masks", data=np.array(store_masks))
        f.create_dataset("predictions", data=np.array(store_preds))
        # Save indices meta-data too
        f.create_dataset("indices", data=selected_indices)
        
    print(f"✅ Saved {out_file}")

if __name__ == "__main__":
    run_visuals()