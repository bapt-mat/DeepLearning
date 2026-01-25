import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
from scipy.ndimage import label
from dataset import ForgeryDataset
from model import FlexibleModel
import kaggle_metric

# --- CONFIGURATION ---
FIXED_PROB_THRESH = 0.50
MIN_AREAS = [0, 5, 10, 25, 50, 75, 100, 200]

def filter_and_encode(binary_mask, min_area):
    """
    Labels connected components and removes those smaller than min_area.
    """
    labeled_mask, num_features = label(binary_mask)
    instances = []
    for k in range(1, num_features + 1):
        instance = (labeled_mask == k).astype(np.uint8)
        if instance.sum() > min_area:
            instances.append(instance)
            
    return "authentic" if not instances else kaggle_metric.rle_encode(instances)

def run_study():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--arch', type=str, default='segformer')
    parser.add_argument('--encoder', type=str, default='mit_b2')
    parser.add_argument('--im_size', type=int, default=256)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📊 Starting Minimum Area Ablation on {args.model_path}")
    print(f"   Fixed Probability Threshold: {FIXED_PROB_THRESH}")
    
    # 1. Load Model
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None).to(device)
    try:
        model.load_state_dict(torch.load(args.model_path, map_location=device))
    except:
        model.load_state_dict(torch.load(args.model_path, map_location=device), strict=False)
    model.eval()
    
    # 2. Load Data
    val_ds = ForgeryDataset(args.data_dir, phase='val', im_size=(args.im_size, args.im_size))
    
    # 3. Pre-calculate Predictions & Build Ground Truth
    print("⚡ Generating base predictions...")
    ground_truth = []
    binary_predictions = [] 
    
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            img, mask = val_ds[i]
            
            # --- FIX IS HERE ---
            # 1. Get shape as a list of integers: [256, 256]
            shape_list = list(mask.squeeze().shape)
            # 2. Convert to string so json.loads() in the metric doesn't crash: "[256, 256]"
            shape_str = str(shape_list)
            
            # GT Encoding
            gt_mask = mask.squeeze().numpy().astype(np.uint8)
            gt_rle = filter_and_encode(gt_mask, min_area=0) 
            
            ground_truth.append({
                'row_id': i, 
                'annotation': gt_rle,
                'shape': shape_str  # <--- Passing String instead of Size object
            })
            
            # Prediction
            input_t = img.unsqueeze(0).to(device)
            out = model(input_t)
            if isinstance(out, list): out = out[0]
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()
            
            pred_bin = (prob_map > FIXED_PROB_THRESH).astype(np.uint8)
            binary_predictions.append(pred_bin)

    sol_df = pd.DataFrame(ground_truth)
    scores = []

    # 4. Test Different Areas
    print("🔄 Testing Minimum Area thresholds...")
    for min_area in MIN_AREAS:
        submission = []
        for i, pred_bin in enumerate(binary_predictions):
            pred_rle = filter_and_encode(pred_bin, min_area)
            submission.append({'row_id': i, 'annotation': pred_rle})
        
        sub_df = pd.DataFrame(submission)
        
        try:
            score = kaggle_metric.score(sol_df, sub_df, 'row_id')
            scores.append(score)
            print(f"   Min Area > {min_area} px: oF1 = {score:.4f}")
        except Exception as e:
            print(f"   ❌ Metric Error at area {min_area}: {e}")
            import traceback
            traceback.print_exc()
            scores.append(0)

    # 5. Plot
    plt.figure(figsize=(10, 6))
    plt.plot(MIN_AREAS, scores, marker='o', linestyle='-', color='purple', linewidth=2)
    plt.title(f"Ablation: Minimum Instance Size (Prob Thresh={FIXED_PROB_THRESH})")
    plt.xlabel("Minimum Area (Pixels)")
    plt.ylabel("oF1 Score")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    if len(scores) > 0 and max(scores) > 0:
        max_score = max(scores)
        max_area = MIN_AREAS[scores.index(max_score)]
        plt.axvline(max_area, color='r', linestyle='--', label=f"Best: >{max_area}px ({max_score:.4f})")
        plt.legend()
    
    plt.savefig("study_min_area.png")
    print("✅ Plot saved to study_min_area.png")

if __name__ == "__main__":
    run_study()