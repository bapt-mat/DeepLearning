import torch
import numpy as np
import pandas as pd
import argparse
import json
import cv2
from tqdm import tqdm
from scipy.ndimage import label, binary_fill_holes
from dataset import ForgeryDataset
from model import FlexibleModel
import kaggle_metric

# --- CONFIGURATION ---
THRESH_CLASS = 0.60  # Gatekeeper: Strict threshold to reject authentic images
THRESH_SEG   = 0.50  # Specialist: Standard threshold for mask generation
MIN_AREA     = 50    # Polisher: Reject objects smaller than 50 pixels

def clean_mask(binary_mask):
    """
    Stage 3: The Polisher
    Refines the raw binary mask to remove noise and fill gaps.
    """
    # 1. Fill Holes (e.g., inside a copy-moved circle)
    # Binary fill holes expects 0/1, so we ensure it's bool, then back to uint8
    mask_filled = binary_fill_holes(binary_mask).astype(np.uint8)
    
    # 2. Morphological Opening (Removes small "salt" noise)
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Area Filtering (Remove tiny distinct objects)
    labeled_mask, num_features = label(mask_clean)
    final_mask = np.zeros_like(mask_clean)
    
    for i in range(1, num_features + 1):
        component = (labeled_mask == i)
        if component.sum() > MIN_AREA:
            final_mask[component] = 1
            
    return final_mask

def mask_to_kaggle_format(binary_mask):
    """Converts a binary mask to the competition string format."""
    labeled_mask, num_features = label(binary_mask)
    instances = []
    for i in range(1, num_features + 1):
        instance = (labeled_mask == i).astype(np.uint8)
        instances.append(instance)
    
    return "authentic" if not instances else kaggle_metric.rle_encode(instances)

def run_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    # We use the SAME model for both stages if it's the best one (B2 Capacity or B2 Aug)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--arch', type=str, default='segformer')
    parser.add_argument('--encoder', type=str, default='mit_b2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Starting FINAL PIPELINE Inference")
    print(f"   Using Model: {args.model_path}")
    print(f"   Gatekeeper Threshold: {THRESH_CLASS}")
    print(f"   Cleaning: Min Area > {MIN_AREA} px")

    # 1. Load Model
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None).to(device)
    # Allow loose loading in case of slight key mismatches
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except:
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()

    # 2. Load Validation Data
    val_ds = ForgeryDataset(args.data_dir, phase='val')
    
    solution = []    # Ground Truth
    submission = []  # Our Predictions
    
    # Statistics
    stats = {"authentic_passed": 0, "authentic_caught": 0, "forged_caught": 0, "forged_missed": 0}

    print(f"⚡ Processing {len(val_ds)} images...")
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            img, mask = val_ds[i]
            gt_mask = mask.squeeze().numpy().astype(np.uint8)
            is_gt_forged = gt_mask.max() > 0
            
            # Prepare Input
            input_t = img.unsqueeze(0).to(device)
            
            # --- INFERENCE ---
            out = model(input_t)
            if isinstance(out, list): out = out[0]
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()
            
            # --- STAGE 1: GATEKEEPER ---
            max_prob = prob_map.max()
            
            if max_prob < THRESH_CLASS:
                # PREDICT AUTHENTIC
                pred_bin = np.zeros_like(gt_mask)
                
                if not is_gt_forged: stats["authentic_passed"] += 1
                else: stats["forged_missed"] += 1
                
            else:
                # PREDICT FORGED -> GO TO STAGE 2
                # --- STAGE 2: SEGMENTATION ---
                raw_mask = (prob_map > THRESH_SEG).astype(np.uint8)
                
                # --- STAGE 3: POLISHER ---
                pred_bin = clean_mask(raw_mask)
                
                # Check if polishing removed everything (rare but possible)
                if pred_bin.max() == 0:
                    # Reverted to Authentic
                    if not is_gt_forged: stats["authentic_passed"] += 1
                    else: stats["forged_missed"] += 1
                else:
                    # Final Forged Prediction
                    if is_gt_forged: stats["forged_caught"] += 1
                    else: stats["authentic_caught"] += 1 # False Positive

            # Save for Scoring
            gt_str = mask_to_kaggle_format(gt_mask)
            pred_str = mask_to_kaggle_format(pred_bin)
            
            solution.append({'row_id': i, 'annotation': gt_str, 'shape': json.dumps(mask.shape[1:])})
            submission.append({'row_id': i, 'annotation': pred_str})

    # 3. Calculate Score
    print("📊 Calculating oF1 Score...")
    score = kaggle_metric.score(pd.DataFrame(solution), pd.DataFrame(submission), 'row_id')
    
    print("\n" + "="*40)
    print(f"🏆 FINAL PIPELINE SCORE: {score:.5f}")
    print("="*40)
    print("Pipeline Stats:")
    print(f"   ✅ Authentic Correctly Ignored: {stats['authentic_passed']}")
    print(f"   ❌ False Alarms (FP):           {stats['authentic_caught']}")
    print(f"   ✅ Forgeries Correctly Found:   {stats['forged_caught']}")
    print(f"   ❌ Forgeries Missed (FN):       {stats['forged_missed']}")
    
    # Save to file
    with open("final_score.txt", "w") as f:
        f.write(f"Score: {score}\nStats: {stats}")

if __name__ == "__main__":
    run_pipeline()