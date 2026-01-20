import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import ForgeryDataset
from model import SimpleUNet
import argparse
import os

# DATA PATH ON CLUSTER
DEFAULT_PATH = "/home_expes/tools/mldm-m2/recodai-luc-scientific-image-forgery-detection"

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default=DEFAULT_PATH)
    parser.add_argument('--epochs', type=int, default=10)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Initialize Datasets
    print(f"Loading data from {args.data_dir}...")
    train_ds = ForgeryDataset(args.data_dir, phase='train')
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=2)
    
    model = SimpleUNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.BCEWithLogitsLoss()
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch} | Batch {i} | Loss: {loss.item():.4f}")
                
        print(f"Epoch {epoch} Finished | Avg Loss: {total_loss/len(train_loader):.4f}")
        
        # Save locally (in the cluster home)
        torch.save(model.state_dict(), f"model_epoch_{epoch}.pth")

if __name__ == "__main__":
    train()