import torch
import cv2
import numpy as np
from model import SimpleUNet
from dataset import ForgeryDataset

# --- SETTINGS ---
# Use the model you downloaded from the cluster
MODEL_PATH = "model_epoch_0.pth" 
# Your local dataset path on your Mac
DATA_PATH = "./data"

# Thresholds
PIXEL_THRESHOLD = 0.5  # A pixel is "white" if prob > 50%
AREA_THRESHOLD = 10    # Image is "Forged" if > 10 white pixels exist

# 1. Load Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SimpleUNet().to(device)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.eval()

# 2. Load One Random Image (Validation Set)
val_ds = ForgeryDataset(DATA_PATH, phase='val')
# Pick a random index or loop to find a specific one
idx = np.random.randint(0, len(val_ds)) 
img_tensor, mask_tensor = val_ds[idx]

print(f"Testing Image Index: {idx}")

# 3. Predict Mask
with torch.no_grad():
    input_batch = img_tensor.unsqueeze(0).to(device)
    output = model(input_batch)
    # Convert logits to probability (0 to 1)
    pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()

# 4. CLASSIFICATION LOGIC (The part you asked about)
# Step A: Binarize the mask (make pixels pure 0 or 1)
pred_mask = (pred_prob > PIXEL_THRESHOLD).astype(np.uint8)

# Step B: Count white pixels
white_pixels = np.sum(pred_mask)

# Step C: Decide Class
if white_pixels > AREA_THRESHOLD:
    prediction_class = "FORGED"
    confidence = pred_prob.max() # How sure is it?
else:
    prediction_class = "AUTHENTIC"
    confidence = 1.0 - pred_prob.max()

# 5. Verify against Ground Truth
gt_is_forged = "FORGED" if mask_tensor.max() > 0 else "AUTHENTIC"

print(f"--- RESULTS ---")
print(f"Ground Truth: {gt_is_forged}")
print(f"Prediction:   {prediction_class}")
print(f"White Pixels: {white_pixels}")
print(f"Confidence:   {confidence:.4f}")

if prediction_class == gt_is_forged:
    print("✅ CORRECT!")
else:
    print("❌ WRONG!")

# 6. Save Visualization
# Prepare images
img_display = (img_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
mask_display = (pred_mask * 255).astype(np.uint8)
gt_display = (mask_tensor.squeeze().numpy() * 255).astype(np.uint8)

# Add text to image
cv2.putText(img_display, f"Pred: {prediction_class}", (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

# Save
cv2.imwrite("final_test_input.png", img_display)
cv2.imwrite("final_test_pred.png", mask_display)
cv2.imwrite("final_test_gt.png", gt_display)
print("\nSaved images: final_test_input.png, final_test_pred.png, final_test_gt.png")