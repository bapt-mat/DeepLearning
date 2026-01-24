import torch
import numpy as np
import pandas as pd
import argparse
import json
import cv2
from tqdm import tqdm
from scipy.ndimage import label, binary_fill_holes
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import ForgeryDataset
from model import FlexibleModel
import kaggle_metric

# --- CONFIGURATION ---
THRESH_CLASS = 0.60  # Gatekeeper threshold
THRESH_SEG   = 0.50  # Segmentation threshold
MIN_AREA     = 50    # Polisher: Min object size

# --- HELPER: PIXEL METRICS ---
def calculate_pixel_metrics(pred_bin, gt_mask):
    """Computes Dice and IoU for a single image."""
    intersection = (pred_bin & gt_mask).sum()
    union = (pred_bin | gt_mask).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    dice = (2. * intersection + 1e-6) / (pred_bin.sum() + gt_mask.sum() + 1e-6)
    return dice, iou

# --- HELPER: CLEANING ---
def clean_mask(binary_mask):
    """Stage 3: The Polisher"""
    # 1. Fill Holes
    mask_filled = binary_fill_holes(binary_mask).astype(np.uint8)
    
    # 2. Morphological Opening
    kernel = np.ones((3, 3), np.uint8)
    mask_clean = cv2.morphologyEx(mask_filled, cv2.MORPH_OPEN, kernel, iterations=1)
    
    # 3. Area Filtering
    labeled_mask, num_features = label(mask_clean)
    final_mask = np.zeros_like(mask_clean)
    for i in range(1, num_features + 1):
        component = (labeled_mask == i)
        if component.sum() > MIN_AREA:
            final_mask[component] = 1
    return final_mask

def mask_to_kaggle_format(binary_mask):
    labeled_mask, num_features = label(binary_mask)
    instances = []
    for i in range(1, num_features + 1):
        instance = (labeled_mask == i).astype(np.uint8)
        instances.append(instance)
    return "authentic" if not instances else kaggle_metric.rle_encode(instances)

def run_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--arch', type=str, default='segformer')
    parser.add_argument('--encoder', type=str, default='mit_b2')
    parser.add_argument('--save_csv', type=str, default='metrics_final_pipeline.csv')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Running FINAL PIPELINE with Full Metrics")
    print(f"   Model: {args.model_path}")

    # 1. Load Model
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except:
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()

    # 2. Data
    val_ds = ForgeryDataset(args.data_dir, phase='val')
    
    # Storage for Metrics
    y_true_cls, y_pred_cls = [], []
    pixel_dices, pixel_ious = [], []
    solution, submission = [], []
    
    print(f"⚡ Processing {len(val_ds)} images...")
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            img, mask = val_ds[i]
            gt_mask = mask.squeeze().numpy().astype(np.uint8)
            is_gt_forged = 1 if gt_mask.max() > 0 else 0
            
            # Inference
            input_t = img.unsqueeze(0).to(device)
            out = model(input_t)
            if isinstance(out, list): out = out[0]
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()
            
            # --- PIPELINE LOGIC ---
            max_prob = prob_map.max()
            
            if max_prob < THRESH_CLASS:
                # Gatekeeper says Authentic
                pred_bin = np.zeros_like(gt_mask)
                is_pred_forged = 0
            else:
                # Gatekeeper says Forged -> Segment & Polish
                raw_mask = (prob_map > THRESH_SEG).astype(np.uint8)
                pred_bin = clean_mask(raw_mask)
                
                # Check if polishing removed everything
                is_pred_forged = 1 if pred_bin.max() > 0 else 0

            # --- METRIC ACCUMULATION ---
            # A. Classification
            y_true_cls.append(is_gt_forged)
            y_pred_cls.append(is_pred_forged)
            
            # B. Segmentation (Only on GT Forged images)
            if is_gt_forged == 1:
                d, iou = calculate_pixel_metrics(pred_bin, gt_mask)
                pixel_dices.append(d)
                pixel_ious.append(iou)

            # C. oF1 Storage
            gt_str = mask_to_kaggle_format(gt_mask)
            pred_str = mask_to_kaggle_format(pred_bin)
            solution.append({'row_id': i, 'annotation': gt_str, 'shape': json.dumps(mask.shape[1:])})
            submission.append({'row_id': i, 'annotation': pred_str})

    # 3. Compute Final Scores
    print("📊 Computing Statistics...")
    
    # Classification
    acc = accuracy_score(y_true_cls, y_pred_cls)
    rec = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    f1_cls = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    
    # Segmentation
    mean_dice = np.mean(pixel_dices) if pixel_dices else 0.0
    mean_iou = np.mean(pixel_ious) if pixel_ious else 0.0
    
    # Official Score
    of1 = kaggle_metric.score(pd.DataFrame(solution), pd.DataFrame(submission), 'row_id')

    # 4. Print & Save
    results = {
        "Model": "Final_Pipeline_B2",
        "oF1": of1,
        "Class_Acc": acc,
        "Class_F1": f1_cls,
        "Class_Recall": rec,
        "Pixel_Dice": mean_dice,
        "Pixel_IoU": mean_iou
    }
    
    print("-" * 40)
    print(f"🏆 FINAL PIPELINE RESULTS:")
    print(f"   🔹 oF1 Score:         {of1:.4f}")
    print(f"   🔹 Classification Acc: {acc:.4f}")
    print(f"   🔹 Classification F1:  {f1_cls:.4f}")
    print(f"   🔹 Pixel Dice:         {mean_dice:.4f}")
    print("-" * 40)
    
    # Save CSV
    df = pd.DataFrame([results])
    df.to_csv(args.save_csv, index=False)
    print(f"✅ Saved metrics to {args.save_csv}")

if __name__ == "__main__":
    run_pipeline()