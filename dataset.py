import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path

class ForgeryDataset(Dataset):
    def __init__(self, data_root, phase='train'):
        self.data_root = Path(data_root)
        self.images = []
        
        # 1. Collect paths
        auth = [(str(p), 0, None) for p in sorted((self.data_root / "train_images" / "authentic").glob("*.png"))]
        forg = []
        mask_root = self.data_root / "train_masks"
        for p in sorted((self.data_root / "train_images" / "forged").glob("*.png")):
            m_path = mask_root / f"{p.stem}.npy"
            forg.append((str(p), 1, str(m_path) if m_path.exists() else None))

        # 2. Stratified Split (80/20)
        def split(data):
            np.random.seed(42)
            perm = np.random.permutation(len(data))
            limit = int(0.8 * len(data))
            return [data[i] for i in perm[:limit]] if phase == 'train' else [data[i] for i in perm[limit:]]

        self.dataset = split(auth) + split(forg)
        if phase == 'train': np.random.shuffle(self.dataset)

    def __len__(self): return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path, label, mask_path = self.dataset[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        mask = np.zeros((h, w), dtype=np.float32)
        if label == 1 and mask_path:
            m = np.load(mask_path)
            if m.ndim == 3: m = m.max(axis=0)
            mask = (m > 0).astype(np.float32)

        img = cv2.resize(img, (256, 256))
        mask = cv2.resize(mask, (256, 256), interpolation=cv2.INTER_NEAREST)
        
        t_img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
        t_mask = torch.from_numpy(mask).float().unsqueeze(0)
        return t_img, t_mask