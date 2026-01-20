import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path

class ForgeryDataset(Dataset):
    def __init__(self, data_root, phase='train', transform=None):
        self.data_root = Path(data_root)
        self.phase = phase
        self.images = []
        self.labels = []
        self.mask_paths = []

        # 1. Scan Authentic
        auth_dir = self.data_root / "train_images" / "authentic"
        if auth_dir.exists():
            for img_path in auth_dir.glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(0)
                self.mask_paths.append(None)

        # 2. Scan Forged
        forg_dir = self.data_root / "train_images" / "forged"
        mask_root = self.data_root / "train_masks"
        if forg_dir.exists():
            for img_path in forg_dir.glob("*.png"):
                self.images.append(str(img_path))
                self.labels.append(1)
                file_id = img_path.stem
                mask_p = mask_root / f"{file_id}.npy"
                self.mask_paths.append(str(mask_p) if mask_p.exists() else None)

        # Split 80/20
        combined = list(zip(self.images, self.labels, self.mask_paths))
        combined.sort(key=lambda x: x[0]) 
        np.random.seed(42)
        indices = np.random.permutation(len(combined))
        split = int(0.8 * len(combined))
        
        if phase == 'train':
            indices = indices[:split]
        else:
            indices = indices[split:]
        self.dataset = [combined[i] for i in indices]

    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path, label, mask_path = self.dataset[idx]
        image = cv2.imread(img_path)
        if image is None: raise FileNotFoundError(f"Missing: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
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

        target_size = (256, 256)
        image = cv2.resize(image, target_size)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)

        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        mask = torch.from_numpy(mask).float().unsqueeze(0)
        return image, mask