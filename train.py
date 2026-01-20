import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ForgeryDataset
from model import SimpleUNet
import pandas as pd
import os
import sys

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")

    # Data
    train_ds = ForgeryDataset(args.data_dir, phase='train')
    val_ds = ForgeryDataset(args.data_dir, phase='val')
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)
    
    # Model
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    history = []
    
    print("🔥 Starting Training Loop...")
    for epoch in range(args.epochs):
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
            
        # Val
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
        
        avg_train = train_loss / len(train_loader)
        avg_val = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:02d} | Train: {avg_train:.4f} | Val: {avg_val:.4f}")
        
        history.append({'epoch': epoch+1, 'train_loss': avg_train, 'val_loss': avg_val})

        # Save model (only latest to save space)
        torch.save(model.state_dict(), "latest_model.pth")

    # Save CSV to disk
    df = pd.DataFrame(history)
    df.to_csv("training_log.csv", index=False)
    
    # PRINT CSV TO STDOUT (Backup for logs)
    print("\n" + "="*40)
    print("FINAL RESULTS (Copy this to Excel)")
    print("="*40)
    print(df.to_csv(index=False))
    print("="*40)

if __name__ == "__main__":
    train()