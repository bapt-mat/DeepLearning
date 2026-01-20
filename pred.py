import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from model import SimpleUNet
from dataset import ForgeryDataset

# --- SETTINGS ---
MODEL_PATH = "model_epoch_3.pth"
DATA_PATH = "./data"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5  # Pixel probability threshold

# Load Model
model = SimpleUNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# Load Data
val_ds = ForgeryDataset(DATA_PATH, phase='val')
print(f"Evaluating on {len(val_ds)} images...")

y_true = []
y_pred = []

with torch.no_grad():
    for img_tensor, mask_tensor in tqdm(val_ds):
        # 1. Get Ground Truth Label
        # If mask has white pixels, it's forged (1). Else authentic (0).
        is_forged_true = 1 if mask_tensor.max() > 0 else 0
        y_true.append(is_forged_true)
        
        # 2. Get Prediction
        input_batch = img_tensor.unsqueeze(0).to(DEVICE)
        output = model(input_batch)
        pred_prob = torch.sigmoid(output)
        
        # 3. Classification Logic
        # If the model finds ANY region with high probability (>0.5), we call it Forged
        # You can tune this: e.g., only if > 10 pixels are detected.
        if pred_prob.max() > THRESHOLD:
            y_pred.append(1) # Predicted Forged
        else:
            y_pred.append(0) # Predicted Authentic

# Calculate Metrics
acc = accuracy_score(y_true, y_pred)
print(f"\n--- Classification Results ---")
print(f"Accuracy: {acc:.4f}")
print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred))
print("\nReport:")
print(classification_report(y_true, y_pred, target_names=["Authentic", "Forged"]))