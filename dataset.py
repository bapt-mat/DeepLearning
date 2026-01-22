import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import glob
import albumentations as A
from albumentations.pytorch import ToTensorV2

class ForgeryDataset(Dataset):
    def __init__(self, root_dir, phase='train', image_size=512):
        self.root_dir = root_dir
        self.phase = phase
        self.image_size = image_size
        
        # Check environment variable to enable heavy augmentation
        # Default is False (Safe for re-training baselines)
        self.use_augmentation = os.environ.get("USE_AUGMENTATION", "False") == "True"

        # Paths
        if self.phase == 'train':
            self.img_dir = os.path.join(root_dir, 'train', 'train')
            self.mask_dir = os.path.join(root_dir, 'train', 'ground_truth')
            self.all_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
            
            # SPLIT: First 80% for training
            limit = int(0.8 * len(self.all_files))
            self.files = self.all_files[:limit]
            
            if self.use_augmentation:
                print("🌪️  DATA AUGMENTATION ENABLED: Flips, Rotations, Shifts Active!")
                self.transform = A.Compose([
                    A.Resize(image_size, image_size),
                    A.HorizontalFlip(p=0.5),
                    A.VerticalFlip(p=0.5),
                    A.RandomRotate90(p=0.5),
                    A.ShiftScaleRotate(scale_limit=0.1, rotate_limit=15, p=0.5),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
            else:
                print("🛡️  Standard Training (No Augmentation)")
                self.transform = A.Compose([
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
            
        else:
            # Validation Phase (Always consistent)
            self.img_dir = os.path.join(root_dir, 'train', 'train')
            self.mask_dir = os.path.join(root_dir, 'train', 'ground_truth')
            self.all_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
            limit = int(0.8 * len(self.all_files))
            self.files = self.all_files[limit:]
            
            self.transform = A.Compose([
                A.Resize(image_size, image_size),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        file_name = os.path.basename(img_path)
        
        # 1. Load Image
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load Mask
        mask_path = os.path.join(self.mask_dir, file_name.replace('.png', '.npy'))
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            if mask.ndim == 3:
                mask = np.max(mask, axis=0)
        else:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        # 3. Apply Transform
        augmented = self.transform(image=image, mask=mask)
        image_tensor = augmented['image']
        mask_tensor = augmented['mask'].float().unsqueeze(0)

        return image_tensor, mask_tensor