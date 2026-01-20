import os
import cv2
import numpy as np
import zipfile
import pandas as pd
from pathlib import Path
from tqdm import tqdm

# --- CONFIGURATION ---
BASE_DIR = Path("data")  # Folder containing 'train_images' and 'train_masks'
OUTPUT_NAME = "dataset"  # Will result in dataset.zip and dataset_chunks/
TARGET_SIZE = (768, 768) # Resize target
CHUNK_SIZE_MB = 45       # GitHub limit

def preprocess_and_split():
    zip_filename = f"{OUTPUT_NAME}.zip"
    data_records = []
    
    print(f"🚀 Starting processing. Target Size: {TARGET_SIZE}")

    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zf:
        
        img_root = BASE_DIR / "train_images"
        mask_root = BASE_DIR / "train_masks"
        
        # We iterate over categories and PREPEND the category to the filename
        # to avoid duplicates like 'images/10.jpg' appearing twice.
        for category in ["authentic", "forged"]:
            folder_path = img_root / category
            if not folder_path.exists(): continue
                
            print(f"📸 Processing {category}...")
            files = list(folder_path.glob("*.png"))
            
            for img_path in tqdm(files):
                file_id = img_path.stem # e.g. "10"
                
                # --- CHANGE IS HERE: Unique Filename ---
                # We save as 'images/authentic_10.jpg' or 'images/forged_10.jpg'
                unique_filename = f"{category}_{file_id}.jpg"
                
                # A. Process Image
                img = cv2.imread(str(img_path))
                if img is None: continue
                
                img_small = cv2.resize(img, TARGET_SIZE, interpolation=cv2.INTER_AREA)
                _, encoded_img = cv2.imencode('.jpg', img_small, [cv2.IMWRITE_JPEG_QUALITY, 90])
                
                # Write with the UNIQUE name
                zf.writestr(f"images/{unique_filename}", encoded_img.tobytes())
                
                # B. Process Mask (Only for Forged)
                has_mask = 0
                mask_filename = "" 
                
                if category == "forged":
                    mask_path = mask_root / f"{file_id}.npy"
                    if mask_path.exists():
                        try:
                            mask = np.load(mask_path)
                            if mask.ndim == 3: mask = mask.max(axis=0)
                            
                            mask = (mask > 0).astype(np.uint8) * 255
                            mask_small = cv2.resize(mask, TARGET_SIZE, interpolation=cv2.INTER_NEAREST)
                            _, encoded_mask = cv2.imencode('.png', mask_small)
                            
                            # Save mask also with unique name to be safe
                            mask_save_name = f"{category}_{file_id}.png"
                            zf.writestr(f"masks/{mask_save_name}", encoded_mask.tobytes())
                            
                            has_mask = 1
                            mask_filename = mask_save_name
                        except:
                            print(f"Error reading mask {file_id}")

                # C. Metadata (Save the exact filename we just created)
                data_records.append({
                    "filename": unique_filename,  # <--- Now we rely on this column
                    "label": 1 if category == "forged" else 0,
                    "has_mask": has_mask,
                    "mask_filename": mask_filename
                })

        # Save CSV inside ZIP
        df = pd.DataFrame(data_records)
        zf.writestr("metadata.csv", df.to_csv(index=False))

    print(f"📦 Created {zip_filename} ({os.path.getsize(zip_filename)/1e6:.1f} MB)")

    # Splitting logic remains the same...
    print("🔪 Splitting into chunks...")
    chunk_dir = Path(f"{OUTPUT_NAME}_chunks")
    chunk_dir.mkdir(exist_ok=True)
    for f in chunk_dir.glob("*"): f.unlink()

    chunk_size = CHUNK_SIZE_MB * 1024 * 1024
    with open(zip_filename, 'rb') as f:
        part_num = 0
        while True:
            chunk = f.read(chunk_size)
            if not chunk: break
            part_name = chunk_dir / f"part_{part_num:03d}"
            with open(part_name, 'wb') as chunk_file:
                chunk_file.write(chunk)
            part_num += 1
            
    print(f"✅ Done! Created {part_num} chunks.")

if __name__ == "__main__":
    preprocess_and_split()