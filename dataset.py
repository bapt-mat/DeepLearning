import torch
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
import glob

# --- SAFE IMPORT BLOCK ---
try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    HAS_ALBUMENTATIONS = True
except ImportError:
    HAS_ALBUMENTATIONS = False
    print("⚠️ Albumentations not found. Using fallback transforms.")

# Fallback class to mimic Albumentations behavior if library is missing
class BasicTransform:
    def __init__(self, size):
        self.size = size
    def __call__(self, image, mask):
        # 1. Resize
        img = cv2.resize(image, (self.size, self.size))
        # Mask needs NEAREST interpolation to keep 0/1 values
        msk = cv2.resize(mask, (self.size, self.size), interpolation=cv2.INTER_NEAREST)
        
        # 2. Normalize (ImageNet stats)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = (img - mean) / std
        
        # 3. ToTensor
        # HWC -> CHW
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
        msk = torch.from_numpy(msk).float().unsqueeze(0)
        
        return {'image': img, 'mask': msk}

class ForgeryDataset(Dataset):
    def __init__(self, root_dir, phase='train', image_size=512):
        self.root_dir = root_dir
        self.phase = phase
        self.image_size = image_size
        
        # Check environment variable
        self.use_augmentation = os.environ.get("USE_AUGMENTATION", "False") == "True"

        # Paths
        if self.phase == 'train':
            self.img_dir = os.path.join(root_dir, 'train', 'train')
            self.mask_dir = os.path.join(root_dir, 'train', 'ground_truth')
            self.all_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
            limit = int(0.8 * len(self.all_files))
            self.files = self.all_files[:limit]
            
            # --- TRANSFORM LOGIC ---
            if self.use_augmentation:
                if not HAS_ALBUMENTATIONS:
                    raise ImportError("🔥 You set USE_AUGMENTATION=True, but 'albumentations' library is missing!")
                
                print("🌪️  DATA AUGMENTATION ENABLED!")
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
                # Standard training (No flips)
                if HAS_ALBUMENTATIONS:
                    self.transform = A.Compose([
                        A.Resize(image_size, image_size),
                        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                        ToTensorV2()
                    ])
                else:
                    self.transform = BasicTransform(image_size)
            
        else:
            # Validation Phase
            self.img_dir = os.path.join(root_dir, 'train', 'train')
            self.mask_dir = os.path.join(root_dir, 'train', 'ground_truth')
            self.all_files = sorted(glob.glob(os.path.join(self.img_dir, "*.png")))
            limit = int(0.8 * len(self.all_files))
            self.files = self.all_files[limit:]
            
            if HAS_ALBUMENTATIONS:
                self.transform = A.Compose([
                    A.Resize(image_size, image_size),
                    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                    ToTensorV2()
                ])
            else:
                self.transform = BasicTransform(image_size)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        file_name = os.path.basename(img_path)
        
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        mask_path = os.path.join(self.mask_dir, file_name.replace('.png', '.npy'))
        if os.path.exists(mask_path):
            mask = np.load(mask_path)
            if mask.ndim == 3: mask = np.max(mask, axis=0)
        else:
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)

        # Apply Transform (Works for both Albumentations and BasicTransform)
        # Note: Albumentations returns dict {'image':...}, BasicTransform does too.
        augmented = self.transform(image=image, mask=mask)
        
        # Albumentations usually returns 'image' directly in the dict
        # But if using BasicTransform, we manually ensured keys match.
        image_tensor = augmented['image']
        
        # Check if mask is already a tensor (BasicTransform) or needs extraction (Albumentations)
        if isinstance(augmented['mask'], torch.Tensor):
            mask_tensor = augmented['mask']
        else:
            mask_tensor = augmented['mask'].float().unsqueeze(0)

        return image_tensor, mask_tensor