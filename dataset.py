import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import os

class ForgeryDataset(Dataset):
    def __init__(self, data_dir, phase='train', transform=None):
        self.data_dir = data_dir
        self.transform = transform
        
        # Read metadata
        csv_path = os.path.join(data_dir, "metadata.csv")
        df = pd.read_csv(csv_path)
        
        # Shuffle deterministically
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        split_point = int(0.8 * len(df))
        
        if phase == 'train':
            self.df = df.iloc[:split_point]
        else:
            self.df = df.iloc[split_point:]
            
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_name = row['filename']
        has_mask = row['has_mask']
        
        # 1. Load Image
        img_path = os.path.join(self.data_dir, "images", img_name)
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Load Mask
        if has_mask:
            mask_name = row['mask_filename']
            mask_path = os.path.join(self.data_dir, "masks", mask_name)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            _, mask = cv2.threshold(mask, 127, 1.0, cv2.THRESH_BINARY)
        else:
            h, w, _ = image.shape
            mask = np.zeros((h, w), dtype=np.float32)

        # 3. Transform (Optional)
        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
            
        # 4. To Tensor
        image = torch.from_numpy(image).float().permute(2, 0, 1) / 255.0
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).float()
            
        return image, mask.unsqueeze(0)