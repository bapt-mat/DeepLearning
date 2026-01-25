import torch
import numpy as np
import argparse
import os
import h5py
from dataset import ForgeryDataset
from model import FlexibleModel

def run_balanced_gen():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_name', type=str, required=True)
    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet34')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Generating balanced visuals (5 Auth / 5 Forged) for: {args.save_name}")

    #loadiing model
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None, n_classes=1).to(device)
    weights_path = f"{args.save_name}.pth"
    if not os.path.exists(weights_path):
        print(f"Error: Weights file '{weights_path}' not found.")
        return
    try:
        model.load_state_dict(torch.load(weights_path, map_location=device))
    except:
        model.load_state_dict(torch.load(weights_path, map_location=device), strict=False)
    model.eval()

    # load dataset
    val_ds = ForgeryDataset(args.data_dir, phase='val')
    
    # Intelligent selection (5 Authentic + 5 Forged)
    print("Scanning dataset for balanced samples")
    
    # Val dataset is a list of tuples: (path, label, mask_path)
    # label 0 = Authentic, 1 = Forged
    auth_indices = [i for i, x in enumerate(val_ds.dataset) if x[1] == 0]
    forg_indices = [i for i, x in enumerate(val_ds.dataset) if x[1] == 1]
    
    if len(auth_indices) < 5 or len(forg_indices) < 5:
        print("Error: Not enough samples in validation set.")
        return

    np.random.seed(42)
    sel_auth = np.random.choice(auth_indices, 5, replace=False)
    sel_forg = np.random.choice(forg_indices, 5, replace=False)
    
    # Combine: First 5 are Authentic, Next 5 are Forged
    selected_indices = np.concatenate([sel_auth, sel_forg])
    
    store_imgs, store_masks, store_preds = [], [], []

    # inference loop
    print(f"Processing 10 balanced samples...")
    with torch.no_grad():
        for idx in selected_indices:
            img, mask = val_ds[idx] 
            
            input_t = img.unsqueeze(0).to(device)
            out = model(input_t)
            if isinstance(out, list): out = out[0]
            
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()
            
            # store
            img_numpy = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask_numpy = mask.squeeze().numpy().astype(np.uint8)
            
            store_imgs.append(img_numpy)
            store_masks.append(mask_numpy)
            store_preds.append(prob_map)

    # save to H5
    out_file = f"balanced_visuals_{args.save_name}.h5"
    with h5py.File(out_file, 'w') as f:
        f.create_dataset("images", data=np.array(store_imgs))
        f.create_dataset("masks", data=np.array(store_masks))
        f.create_dataset("predictions", data=np.array(store_preds))
        
    print(f"Saved {out_file}")

if __name__ == "__main__":
    run_balanced_gen()