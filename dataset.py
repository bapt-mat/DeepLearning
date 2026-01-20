import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path

class ForgeryDataset(Dataset):
    def __init__(self, data_root, phase='train', transform=None):
        self.data_root = Path(data_root)
        self.phase = phase
        self.transform = transform
        
        # Temporary lists to hold split data
        self.images = []
        self.labels = []
        self.mask_paths = []

        # --- 1. GATHER ALL DATA FIRST ---
        all_authentic = []
        all_forged = []

        # Scan Authentic
        auth_dir = self.data_root / "train_images" / "authentic"
        if auth_dir.exists():
            for img_path in sorted(auth_dir.glob("*.png")):
                # Store tuple: (path, label, mask_path)
                all_authentic.append((str(img_path), 0, None))

        # Scan Forged
        forg_dir = self.data_root / "train_images" / "forged"
        mask_root = self.data_root / "train_masks"
        if forg_dir.exists():
            for img_path in sorted(forg_dir.glob("*.png")):
                file_id = img_path.stem
                mask_p = mask_root / f"{file_id}.npy"
                mask_path_str = str(mask_p) if mask_p.exists() else None
                all_forged.append((str(img_path), 1, mask_path_str))

        # --- 2. STRATIFIED SPLIT ---
        # We split Authentic and Forged separately to ensure balanced classes
        
        def get_split_indices(data_list, phase, split_ratio=0.8):
            np.random.seed(42) # Deterministic shuffle
            indices = np.random.permutation(len(data_list))
            split_point = int(split_ratio * len(data_list))
            
            if phase == 'train':
                return [data_list[i] for i in indices[:split_point]]
            else:
                return [data_list[i] for i in indices[split_point:]]

        # Get the subsets for this phase
        auth_subset = get_split_indices(all_authentic, phase)
        forg_subset = get_split_indices(all_forged, phase)

        # Combine them
        self.dataset = auth_subset + forg_subset
        
        # Optional: Shuffle the combined dataset so batches are mixed
        if phase == 'train':
            np.random.shuffle(self.dataset)

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path, label, mask_path = self.dataset[idx]
        
        # Load Image
        image = cv2.imread(img_path)
        if image is None: 
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load Mask
        h, w, _ = image.shape
        if label == 1 and mask_path:
            try:
                mask = np.load(mask_path)
                if mask.ndim == 3: mask = mask.max(axis=0)
                mask = (mask > 0).astype(np.float32)
            except:
                mask = np.zeros((h, w), dtype=np.float32)
        else:
            mask = np.zeros((h, w), dtype=np.float32)

        # Resize (256x256)
        target_size = (256, 256)
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        # To Tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)
            
        return image, mask