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
    print(f"📂 Dataset loaded. Finding forged samples...")

    # 3. Find Forged Images (Label == 1)
    # We want to find exactly 5 forged images to visualize
    found_count = 0
    indices_to_save = []
    
    # Scan through the dataset until we find 5 forgeries
    for i in range(len(val_ds)):
        # We peek at the internal list of the dataset to check the label without loading the image
        # val_ds.images is a list of tuples: (image_path, label, mask_path)
        _, label, _ = val_ds.images[i]
        
        if label == 1:
            indices_to_save.append(i)
            found_count += 1
            if found_count >= 5: # Stop after finding 5
                break
    
    if found_count == 0:
        print("⚠️ Warning: No forged images found in the validation set!")
        return

    # 4. Save Predictions
    with h5py.File(f"visuals_{args.save_name}.h5", "w") as f:
        print(f"📸 Saving 5 forged samples to visuals_{args.save_name}.h5...")
        
        for i, idx in enumerate(indices_to_save):
            img_tensor, mask_tensor = val_ds[idx]
            
            # Predict
            with torch.no_grad():
                input_t = img_tensor.unsqueeze(0).to(device)
                out = model(input_t)
                if isinstance(out, list): out = out[0]
                pred_prob = torch.sigmoid(out).squeeze().cpu().numpy()
            
            # Prepare data for saving
            img_np = img_tensor.permute(1, 2, 0).numpy() # (C,H,W) -> (H,W,C)
            mask_np = mask_tensor.squeeze().numpy()
            
            # Save
            grp = f.create_group(f"sample_{i}")
            grp.create_dataset("image", data=img_np)
            grp.create_dataset("gt", data=mask_np)
            grp.create_dataset("pred", data=pred_prob)
            
    print("✅ Done!")

if __name__ == "__main__":
    save_visuals()