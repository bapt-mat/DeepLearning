import torch
import numpy as np
import pandas as pd
import json
from scipy.ndimage import label
from tqdm import tqdm
from model import ResUNet  # Change to SimpleUNet if using the old model
from dataset import ForgeryDataset
import kaggle_metric  # This imports the file you just created

# --- SETTINGS ---
MODEL_PATH = "model_epoch_19.pth"  # Path to your best model
DATA_PATH = "./data" # Path to data
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5  # Probability threshold to decide pixel is white
MIN_PIXELS = 10  # Minimum size to count as a "forgery instance"

def mask_to_kaggle_format(binary_mask):
    """
    Converts a binary mask (H, W) into the format expected by kaggle_metric.py:
    - If empty: returns "authentic"
    - If forged: returns RLE string of distinct instances
    """
    # 1. Instance Recovery (Connected Components)
    # The metric requires a LIST of binary masks (one per forged region).
    # Since your model outputs one big mask, we separate blobs using 'label'.
    labeled_mask, num_features = label(binary_mask)
    
    instances = []
    for i in range(1, num_features + 1):
        instance_mask = (labeled_mask == i).astype(np.uint8)
        # Optional: Filter out tiny noise artifacts
        if instance_mask.sum() > MIN_PIXELS:
            instances.append(instance_mask)
            
    # 2. Return formatted string
    if not instances:
        return "authentic"
    else:
        # Use the official encoder you provided
        return kaggle_metric.rle_encode(instances)

def run_evaluation():
    print(f"Loading model from {MODEL_PATH}...")
    model = ResUNet().to(DEVICE) 
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print(f"Loading validation data from {DATA_PATH}...")
    val_ds = ForgeryDataset(DATA_PATH, phase='val')
    
    # Lists to build DataFrames
    solution_rows = []    # Ground Truth
    submission_rows = []  # Predictions
    
    print("Running inference and formatting...")
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            img_tensor, mask_tensor = val_ds[i]
            
            # --- 1. Ground Truth Processing ---
            gt_mask = mask_tensor.squeeze().numpy().astype(np.uint8)
            height, width = gt_mask.shape
            
            # Get Ground Truth String (RLE or 'authentic')
            # We use the same 'label' logic on GT to ensure instances are defined
            gt_annotation = mask_to_kaggle_format(gt_mask)
            
            # Shape string is required by the metric
            shape_str = json.dumps([height, width])
            
            # --- 2. Prediction Processing ---
            input_batch = img_tensor.unsqueeze(0).to(DEVICE)
            output = model(input_batch)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_bin = (pred_prob > THRESHOLD).astype(np.uint8)
            
            # Get Prediction String
            pred_annotation = mask_to_kaggle_format(pred_bin)
            
            # --- 3. Store Data ---
            # We use 'i' as the row_id
            solution_rows.append({
                'row_id': i, 
                'annotation': gt_annotation, 
                'shape': shape_str
            })
            submission_rows.append({
                'row_id': i, 
                'annotation': pred_annotation
            })

    # --- 4. Create DataFrames ---
    solution_df = pd.DataFrame(solution_rows)
    submission_df = pd.DataFrame(submission_rows)

    print("\nComputing Official Score...")
    try:
        # Call the official score function
        final_score = kaggle_metric.score(solution_df, submission_df, 'row_id')
        
        print("\n" + "="*40)
        print(f"✅ OFFICIAL oF1 SCORE: {final_score:.5f}")
        print("="*40)
        
    except Exception as e:
        print(f"\n❌ Error in metric calculation: {e}")
        print("Make sure 'numba' is installed: pip install numba")

if __name__ == "__main__":
    run_evaluation()