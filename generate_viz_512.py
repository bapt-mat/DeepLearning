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
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💾 Generating H5 visuals for: {args.save_name}")

    # 1. Load Model
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None, n_classes=1).to(device)
    
    weights_path = f"{args.save_name}.pth"
    if not os.path.exists(weights_path):
        print(f"❌ Error: Weights file '{weights_path}' not found.")
        return

    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except:
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
        
    model.eval()

    # 2. Load Dataset
    val_ds = ForgeryDataset(args.data_dir, phase='val', im_size=(512, 512))
    
    # 3. Find Forged Samples (Intelligent Scanning)
    print("🔍 Scanning for forged validation samples...")
    forged_indices = [i for i, x in enumerate(val_ds.dataset) if x[1] == 1]
    
    if len(forged_indices) == 0:
        print("❌ Error: No forged images found in validation set.")
        return

    # Select 10 random samples to save
    np.random.seed(42)
    count = min(len(forged_indices), 10)
    selected_indices = np.random.choice(forged_indices, count, replace=False)
    
    # Lists to store data
    store_imgs = []
    store_masks = []
    store_preds = []

    # 4. Inference Loop
    print(f"⚡ Processing {count} samples...")
    with torch.no_grad():
        for idx in selected_indices:
            img, mask = val_ds[idx] # img is (3, H, W) normalized 0-1
            
            input_t = img.unsqueeze(0).to(device)
            out = model(input_t)
            if isinstance(out, list): out = out[0]
            
            # Probability Map (0-1)
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()
            
            # Prepare for storage
            # Convert image back to 0-255 uint8 for space efficiency
            img_numpy = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask_numpy = mask.squeeze().numpy().astype(np.uint8)
            
            store_imgs.append(img_numpy)
            store_masks.append(mask_numpy)
            store_preds.append(prob_map)

    # 5. Save to H5
    out_file = f"visuals_{args.save_name}.h5"
    with h5py.File(out_file, 'w') as f:
        f.create_dataset("images", data=np.array(store_imgs))
        f.create_dataset("masks", data=np.array(store_masks))
        f.create_dataset("predictions", data=np.array(store_preds))
        
    print(f"✅ Saved {out_file} (Size: {os.path.getsize(out_file)/1024/1024:.2f} MB)")

if __name__ == "__main__":
    run_visuals()