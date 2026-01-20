import torch
import numpy as np
import pandas as pd
import json
from scipy.ndimage import label
from tqdm import tqdm
from model import SimpleUNet
from dataset import ForgeryDataset
import kaggle_metric  # Import the file you created above

# --- SETTINGS ---
MODEL_PATH = "model_epoch_3.pth"  # Your trained model
DATA_PATH = "./data" # Dataset path
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5  # Probability threshold
MIN_PIXELS = 10  # Ignore instances smaller than this (noise reduction)

def get_rle_string_from_mask(binary_mask):
    """
    Takes a binary mask (H,W), separates it into instances, 
    and returns the specific 'authentic' string or RLE string 
    expected by the kaggle_metric.py
    """
    # 1. Connected Components (Instance Separation)
    labeled_mask, num_features = label(binary_mask)
    
    instances = []
    for i in range(1, num_features + 1):
        instance = (labeled_mask == i).astype(np.uint8)
        # Filter noise
        if instance.sum() > MIN_PIXELS:
            instances.append(instance)
            
    # 2. Convert to Kaggle Format
    if len(instances) == 0:
        return "authentic"
    else:
        # Use the official RLE encoder
        return kaggle_metric.rle_encode(instances)

def evaluate_official():
    print(f"Loading model: {MODEL_PATH}")
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    print("Loading Validation Set...")
    val_ds = ForgeryDataset(DATA_PATH, phase='val')
    
    solution_rows = []
    submission_rows = []
    
    print("Generating Predictions & RLEs...")
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            img_tensor, gt_tensor = val_ds[i]
            
            # --- PREDICTION ---
            input_batch = img_tensor.unsqueeze(0).to(DEVICE)
            output = model(input_batch)
            pred_prob = torch.sigmoid(output).squeeze().cpu().numpy()
            pred_bin = (pred_prob > THRESHOLD).astype(np.uint8)
            
            # --- GROUND TRUTH ---
            gt_bin = gt_tensor.squeeze().numpy().astype(np.uint8)
            
            # --- FORMATTING FOR OFFICIAL METRIC ---
            # 1. Get Shape JSON
            height, width = pred_bin.shape
            shape_str = json.dumps([height, width])
            
            # 2. Get Prediction String (RLE or 'authentic')
            pred_str = get_rle_string_from_mask(pred_bin)
            
            # 3. Get Solution String (RLE or 'authentic')
            # Note: Even though we collapsed the mask in dataset.py, 
            # we must treat connected components as instances for the metric.
            gt_str = get_rle_string_from_mask(gt_bin)
            
            # 4. Append to lists
            row_id = i
            solution_rows.append({'row_id': row_id, 'annotation': gt_str, 'shape': shape_str})
            submission_rows.append({'row_id': row_id, 'annotation': pred_str})

    # --- DATAFRAME CREATION ---
    solution_df = pd.DataFrame(solution_rows)
    submission_df = pd.DataFrame(submission_rows)
    
    print("\nCalculating Final Score...")
    try:
        final_score = kaggle_metric.score(solution_df, submission_df, 'row_id')
        print("=" * 40)
        print(f"OFFICIAL KAGGLE SCORE (oF1): {final_score:.4f}")
        print("=" * 40)
    except Exception as e:
        print(f"Error during scoring: {e}")
        print("Tip: Make sure you installed 'numba': pip install numba")

if __name__ == "__main__":
    evaluate_official()