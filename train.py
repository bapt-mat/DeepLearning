import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ForgeryDataset
from model import SimpleUNet
from metrics import calculate_metrics  # Import the new metrics file
import argparse
import os

DEFAULT_PATH = "/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DEFAULT_PATH)
    parser.add_argument('--epochs', type=int, default=20)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 1. Load Datasets (Train & Val)
    train_ds = ForgeryDataset(args.data_dir, phase='train')
    val_ds = ForgeryDataset(args.data_dir, phase='val')
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False, num_workers=2)
    
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(args.epochs):
        # --- TRAINING LOOP ---
        model.train()
        train_loss = 0
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # --- VALIDATION LOOP ---
        model.eval()
        val_loss = 0
        val_dice = 0
        val_iou = 0
        
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_loss += criterion(outputs, masks).item()
                
                # Calculate metrics for this batch
                batch_metrics = calculate_metrics(outputs, masks)
                val_dice += batch_metrics["Dice"]
                val_iou += batch_metrics["IoU"]
        
        # Averages
        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        avg_dice = val_dice / len(val_loader)
        avg_iou = val_iou / len(val_loader)
        
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {avg_val_loss:.4f} | "
              f"Dice: {avg_dice:.4f} | IoU: {avg_iou:.4f}")
        
        # Save Model
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()