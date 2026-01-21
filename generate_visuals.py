import torch
import h5py
import argparse
import numpy as np
import cv2
from dataset import ForgeryDataset
from model import FlexibleModel

def save_visuals():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_name', type=str, default='unet_baseline')
    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet34')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None).to(device)
    try:
        model.load_state_dict(torch.load(f"{args.save_name}.pth", map_location=device))
        print(f"Loaded {args.save_name}.pth")
    except:
        print(f"❌ Could not load weights for {args.save_name}!")
        return

    model.eval()
    val_ds = ForgeryDataset(args.data_dir, phase='val')
    
    # Select 5 specific indices (hardcoded to ensure we get forgeries)
    # You can change these or pick random ones
    indices = [0, 10, 25, 42, 55] 
    
    with h5py.File(f"visuals_{args.save_name}.h5", "w") as f:
        print("📸 Saving visuals...")
        
        for i, idx in enumerate(indices):
            if idx >= len(val_ds): continue
            
            img_tensor, mask_tensor = val_ds[idx]
            
            # Predict
            with torch.no_grad():
                input_t = img_tensor.unsqueeze(0).to(device)
                out = model(input_t)
                if isinstance(out, list): out = out[0]
                pred_prob = torch.sigmoid(out).squeeze().cpu().numpy()
            
            # Convert back to standard image format for saving
            # Image: (3, 256, 256) -> (256, 256, 3)
            img_np = img_tensor.permute(1, 2, 0).numpy()
            mask_np = mask_tensor.squeeze().numpy()
            
            # Create group for this image
            grp = f.create_group(f"sample_{i}")
            grp.create_dataset("image", data=img_np)
            grp.create_dataset("gt", data=mask_np)
            grp.create_dataset("pred", data=pred_prob)
            
    print(f"✅ Saved visuals_{args.save_name}.h5. Download this file!")

if __name__ == "__main__":
    save_visuals()