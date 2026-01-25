import torch
import numpy as np
import h5py
import os
from dataset import ForgeryDataset
from model import FlexibleModel

# --- CONFIGURATION ---
DATA_DIR = "/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"
MODEL_PATH = "segformer_b2_512.pth"
OUTPUT_FILE = "visuals_segformer_b2_512.h5"
IM_SIZE = 512

def run():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💾 Generating 512x512 Visuals from {MODEL_PATH}...")

    # 1. Load Model (SegFormer B2)
    model = FlexibleModel(arch="segformer", encoder="mit_b2", weights=None, n_classes=1).to(device)
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Error: Model file '{MODEL_PATH}' not found!")
        return

    # Load weights
    try:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    except:
        print("⚠️  Loading with strict=False")
        model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    model.eval()

    # 2. Load Validation Dataset at 512x512
    # Crucial: Passing the tuple (512, 512) ensures images are loaded correctly
    val_ds = ForgeryDataset(DATA_DIR, phase='val', im_size=(IM_SIZE, IM_SIZE))
    
    # 3. Find 10 Forged Images
    print("🔍 Scanning validation set for forged examples...")
    indices = []
    
    # Simple scan: Find first 10 images that are actually forged (GT > 0)
    for i in range(len(val_ds)):
        _, mask = val_ds[i]
        if mask.max() > 0: 
            indices.append(i)
        if len(indices) >= 10: 
            break
            
    if not indices:
        print("❌ No forged images found.")
        return

    # 4. Run Inference
    store_imgs = []
    store_masks = []
    store_preds = []
    
    print(f"⚡ Processing {len(indices)} samples at {IM_SIZE}x{IM_SIZE}...")
    with torch.no_grad():
        for idx in indices:
            img, mask = val_ds[idx]
            
            # Prediction
            input_t = img.unsqueeze(0).to(device)
            out = model(input_t)
            if isinstance(out, list): out = out[0]
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()
            
            # Storage (Convert image back to 0-255 uint8 for space)
            img_numpy = (img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            mask_numpy = mask.squeeze().numpy().astype(np.uint8)
            
            store_imgs.append(img_numpy)
            store_masks.append(mask_numpy)
            store_preds.append(prob_map)

    # 5. Save to H5
    with h5py.File(OUTPUT_FILE, 'w') as f:
        f.create_dataset("images", data=np.array(store_imgs))
        f.create_dataset("masks", data=np.array(store_masks))
        f.create_dataset("predictions", data=np.array(store_preds))
        f.create_dataset("indices", data=indices)

    print(f"✅ Saved {OUTPUT_FILE}")

if __name__ == "__main__":
    run()