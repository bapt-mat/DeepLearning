import torch
import numpy as np
import pandas as pd
import argparse
import json
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from dataset import ForgeryDataset
from model import FlexibleModel
from scipy.ndimage import label
import kaggle_metric

# --- HELPER: RLE ENCODING FOR oF1 ---
def mask_to_kaggle_format(binary_mask):
    labeled_mask, num_features = label(binary_mask)
    instances = []
    for i in range(1, num_features + 1):
        instance = (labeled_mask == i).astype(np.uint8)
        # Filter tiny instances (noise)
        if instance.sum() > 10: 
            instances.append(instance)
    return "authentic" if not instances else kaggle_metric.rle_encode(instances)

# --- HELPER: PIXEL METRICS ---
def calculate_pixel_metrics(pred_bin, gt_mask):
    # Intersection and Union for IoU
    intersection = (pred_bin & gt_mask).sum()
    union = (pred_bin | gt_mask).sum()
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Dice
    dice = (2. * intersection + 1e-6) / (pred_bin.sum() + gt_mask.sum() + 1e-6)
    return dice, iou

def evaluate():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--save_name', type=str, required=True)
    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet34')
    # NEW ARGUMENT: Image Size
    parser.add_argument('--im_size', type=int, default=256, help="Resolution used for training (e.g., 256 or 512)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Evaluating full metrics for: {args.save_name} at {args.im_size}x{args.im_size}")
    
    # 1. Load Model
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None).to(device)
    try:
        model.load_state_dict(torch.load(f"{args.save_name}.pth", map_location=device))
    except FileNotFoundError:
        print(f"❌ Weights not found for {args.save_name}")
        return
    model.eval()

    # 2. Dataset (Pass the size tuple)
    val_ds = ForgeryDataset(args.data_dir, phase='val', im_size=(args.im_size, args.im_size))
    
    # 3. Storage
    # Classification (Image Level)
    y_true_cls, y_pred_cls = [], []
    # Segmentation (Pixel Level)
    pixel_dices, pixel_ious = [], []
    # oF1 (Instance Level)
    solution, submission = [], []

    print("⚡ Running inference...")
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            img, mask = val_ds[i] # mask is tensor (1, H, W)
            
            # Ground Truth Prep
            gt_mask = mask.squeeze().numpy().astype(np.uint8) # (H, W)
            is_forged_gt = 1 if gt_mask.max() > 0 else 0
            
            # Predict
            input_t = img.unsqueeze(0).to(device)
            out = model(input_t)
            if isinstance(out, list): out = out[0]
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy() # (H, W)
            
            # Binarize
            pred_bin = (prob_map > 0.5).astype(np.uint8)
            is_forged_pred = 1 if pred_bin.max() > 0 else 0
            
            # --- A. Classification Metrics ---
            y_true_cls.append(is_forged_gt)
            y_pred_cls.append(is_forged_pred)
            
            # --- B. Segmentation Metrics (Only for forged images to avoid bias) ---
            if is_forged_gt == 1:
                d, iou = calculate_pixel_metrics(pred_bin, gt_mask)
                pixel_dices.append(d)
                pixel_ious.append(iou)
            
            # --- C. oF1 Metrics ---
            gt_str = mask_to_kaggle_format(gt_mask)
            pred_str = mask_to_kaggle_format(pred_bin)
            solution.append({'row_id': i, 'annotation': gt_str, 'shape': json.dumps(mask.shape[1:])})
            submission.append({'row_id': i, 'annotation': pred_str})

    # 4. Compute Final Scores
    # Classification
    acc = accuracy_score(y_true_cls, y_pred_cls)
    prec = precision_score(y_true_cls, y_pred_cls, zero_division=0)
    rec = recall_score(y_true_cls, y_pred_cls, zero_division=0)
    f1_cls = f1_score(y_true_cls, y_pred_cls, zero_division=0)
    
    # Segmentation
    mean_dice = np.mean(pixel_dices) if pixel_dices else 0.0
    mean_iou = np.mean(pixel_ious) if pixel_ious else 0.0
    
    # oF1
    of1 = kaggle_metric.score(pd.DataFrame(solution), pd.DataFrame(submission), 'row_id')
    
    # 5. Print & Save
    results = {
        "Model": args.save_name,
        "oF1": of1,
        "Class_Acc": acc,
        "Class_F1": f1_cls,
        "Class_Recall": rec,
        "Pixel_Dice": mean_dice,
        "Pixel_IoU": mean_iou
    }
    
    print("-" * 40)
    print(f"🏆 RESULTS for {args.save_name}:")
    print(f"   🔹 oF1 Score (Official): {of1:.4f}")
    print(f"   🔹 Classification Acc:   {acc:.4f}")
    print(f"   🔹 Classification F1:    {f1_cls:.4f}")
    print(f"   🔹 Segmentation Dice:    {mean_dice:.4f}")
    print(f"   🔹 Segmentation IoU:     {mean_iou:.4f}")
    print("-" * 40)
    
    # Save to CSV line
    df = pd.DataFrame([results])
    output_file = f"metrics_{args.save_name}.csv"
    df.to_csv(output_file, index=False)
    print(f"✅ Saved metrics to {output_file}")

if __name__ == "__main__":
    evaluate()