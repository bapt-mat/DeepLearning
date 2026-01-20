import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import argparse
import numpy as np
from dataset import ForgeryDataset
from model import SimpleUNet  # CHANGED: Using Standard U-Net
from metrics import calculate_metrics

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data
    train_ds = ForgeryDataset(args.data_dir, phase='train')
    val_ds = ForgeryDataset(args.data_dir, phase='val')
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    
    # Model (Standard U-Net Baseline)
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    # History Storage
    history = {"train_loss": [], "val_loss": [], "val_dice": [], "val_iou": []}

    print("🚀 Starting Training (Baseline U-Net)...")
    for epoch in range(args.epochs):
        # 1. Train
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # 2. Validate
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
                metrics = calculate_metrics(outputs, masks)
                val_dice += metrics["Dice"]
                val_iou += metrics["IoU"]

        # 3. Averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        avg_iou = val_iou / len(val_loader)

        # 4. Store
        history["train_loss"].append(avg_train_loss)
        history["val_loss"].append(avg_val_loss)
        history["val_dice"].append(avg_dice)
        history["val_iou"].append(avg_iou)

        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {avg_train_loss:.4f} | Val Dice: {avg_dice:.4f}")

        # 5. Save History to .h5
        with h5py.File("training_history.h5", "w") as f:
            f.create_dataset("train_loss", data=history["train_loss"])
            f.create_dataset("val_loss", data=history["val_loss"])
            f.create_dataset("val_dice", data=history["val_dice"])
            f.create_dataset("val_iou", data=history["val_iou"])
            f.create_dataset("epochs", data=np.arange(1, epoch + 2))
        
        # 6. Save Model
        torch.save(model.state_dict(), f"model_baseline.pth")

if __name__ == "__main__":
    train()