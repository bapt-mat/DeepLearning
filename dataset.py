import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
from pathlib import Path
import os

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False

class ForgeryDataset(Dataset):
    def __init__(self, data_root, phase='train', im_size=(256, 256)):
        self.data_root = Path(data_root)
        self.phase = phase
        self.im_size = im_size
        self.images = []
        
        # Check environment variable for Augmentation
        self.use_augmentation = os.environ.get("USE_AUGMENTATION", "False") == "True"

        auth = [(str(p), 0, None) for p in sorted((self.data_root / "train_images" / "authentic").glob("*.png"))]
        forg = []
        mask_root = self.data_root / "train_masks"
        for p in sorted((self.data_root / "train_images" / "forged").glob("*.png")):
            m_path = mask_root / f"{p.stem}.npy"
            forg.append((str(p), 1, str(m_path) if m_path.exists() else None))

        # Stratified Split (80% Train, 20% Val)
        def split(data):
            np.random.seed(42) 
            perm = np.random.permutation(len(data))
            limit = int(0.8 * len(data))
            if phase == 'train':
                return [data[i] for i in perm[:limit]]
            else: 
                return [data[i] for i in perm[limit:]]

        self.dataset = split(auth) + split(forg)
        
        if phase == 'train': 
            np.random.shuffle(self.dataset)

        # albumentations Transformations
        if self.use_augmentation and self.phase == 'train':
            if not HAS_ALBUMENTATIONS:
                raise ImportError("USE_AUGMENTATION=True but albumentations problem")
            
            print("data augmentation oui")
            self.transform = A.Compose([
                A.Resize(self.im_size[0], self.im_size[1]),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomRotate90(p=0.5),
                A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, p=0.5),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])
        else:
            self.transform = None

    def __len__(self): return len(self.dataset)
    
    def __getitem__(self, idx):
        img_path, label, mask_path = self.dataset[idx]
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        h, w, _ = img.shape
        if label == 1 and mask_path:
            m = np.load(mask_path)
            if m.ndim == 3: m = m.max(axis=0)
            mask = (m > 0).astype(np.float32)
        else:
            mask = np.zeros((h, w), dtype=np.float32)

        if self.transform is not None:
            augmented = self.transform(image=img, mask=mask)
            t_img = augmented['image']
            t_mask = augmented['mask'].float().unsqueeze(0)
        else:
            img = cv2.resize(img, self.im_size)
            mask = cv2.resize(mask, self.im_size, interpolation=cv2.INTER_NEAREST)
            
            t_img = torch.from_numpy(img).float().permute(2, 0, 1) / 255.0
            t_mask = torch.from_numpy(mask).float().unsqueeze(0)
            
        return t_img, t_mask