import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import h5py
import argparse
import numpy as np
from dataset import ForgeryDataset
from model import FlexibleModel

# --- METRICS ---
def calculate_metrics(pred, target):
    pred = (torch.sigmoid(pred) > 0.5).float()
    tp, fp, fn = (pred * target).sum(), (pred * (1-target)).sum(), ((1-pred) * target).sum()
    # FIXED: Added .item() to ensure it returns a Python float, not a Tensor
    score = (2*tp + 1e-6)/(2*tp + fp + fn + 1e-6)
    return {"Dice": score.item()}

# --- DICE LOSS ---
class DiceLoss(torch.nn.Module):
    def __init__(self): super().__init__()
    def forward(self, inputs, targets, smooth=1):
        inputs = torch.sigmoid(inputs).view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()
        return 1 - (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)

def train():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--weights', type=str, default='imagenet')
    parser.add_argument('--loss', type=str, default='bce')
    parser.add_argument('--save_name', type=str, default='model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Config: {args.arch} | {args.encoder} | {args.weights} | {args.loss}")

    # Data
    train_loader = DataLoader(ForgeryDataset(args.data_dir, phase='train'), batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(ForgeryDataset(args.data_dir, phase='val'), batch_size=16, shuffle=False, num_workers=2)
    
    # Model
    weights_val = None if args.weights == 'None' else args.weights
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=weights_val).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    criterion = DiceLoss() if args.loss == 'dice' else torch.nn.BCEWithLogitsLoss()
    
    history = {"train_loss": [], "val_dice": []}

    for epoch in range(args.epochs):
        model.train()
        train_loss = 0
        for imgs, masks in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            
            if isinstance(outputs, list):
                loss = 0
                for i, out in enumerate(outputs):
                    target_resized = torch.nn.functional.interpolate(masks, size=out.shape[2:], mode='nearest')
                    loss += (1.0 if i == 0 else 0.5) * criterion(out, target_resized)
            else:
                loss = criterion(outputs, masks)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validation
        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                if isinstance(outputs, list): outputs = outputs[0]
                val_dice += calculate_metrics(outputs, masks)["Dice"]

        # Ensure these are floats before saving
        avg_loss = float(train_loss / len(train_loader))
        avg_dice = float(val_dice / len(val_loader))
        
        history["train_loss"].append(avg_loss)
        history["val_dice"].append(avg_dice)
        
        print(f"Epoch {epoch+1} | Loss: {avg_loss:.4f} | Dice: {avg_dice:.4f}")

        # Save History
        with h5py.File(f"results_{args.save_name}.h5", "w") as f:
            f.create_dataset("train_loss", data=history["train_loss"])
            f.create_dataset("val_dice", data=history["val_dice"])
            f.create_dataset("epochs", data=np.arange(1, epoch + 2))
        
        torch.save(model.state_dict(), f"{args.save_name}.pth")

if __name__ == "__main__":
    train()