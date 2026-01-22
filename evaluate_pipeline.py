import torch
import numpy as np
import pandas as pd
import json
import argparse
from tqdm import tqdm
from dataset import ForgeryDataset
from model import FlexibleModel
import kaggle_metric
from scipy.ndimage import label

# --- HELPER: RLE ENCODING ---
def mask_to_kaggle_format(binary_mask):
    labeled_mask, num_features = label(binary_mask)
    instances = []
    for i in range(1, num_features + 1):
        instance = (labeled_mask == i).astype(np.uint8)
        # Filter small noise (optional but recommended)
        if instance.sum() > 10: instances.append(instance)
    return "authentic" if not instances else kaggle_metric.rle_encode(instances)

def evaluate_pipeline():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    # Model 1: The Classifier (Gatekeeper)
    parser.add_argument('--cls_model', type=str, default='segformer_b2_capacity')
    parser.add_argument('--cls_arch', type=str, default='segformer')
    parser.add_argument('--cls_enc', type=str, default='mit_b2')
    # Model 2: The Segmenter (Artist)
    parser.add_argument('--seg_model', type=str, default='unet_baseline')
    parser.add_argument('--seg_arch', type=str, default='unet')
    parser.add_argument('--seg_enc', type=str, default='resnet34')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Launching Pipeline Evaluation")
    print(f"   🛡️  Gatekeeper (Classifier): {args.cls_model}")
    print(f"   🎨 Artist     (Segmenter):  {args.seg_model}")

    # 1. Load Classifier Model
    print("   ... Loading Classifier weights")
    model_cls = FlexibleModel(arch=args.cls_arch, encoder=args.cls_enc, weights=None).to(device)
    model_cls.load_state_dict(torch.load(f"{args.cls_model}.pth", map_location=device))
    model_cls.eval()

    # 2. Load Segmentation Model
    print("   ... Loading Segmenter weights")
    model_seg = FlexibleModel(arch=args.seg_arch, encoder=args.seg_enc, weights=None).to(device)
    model_seg.load_state_dict(torch.load(f"{args.seg_model}.pth", map_location=device))
    model_seg.eval()

    # 3. Dataset
    val_ds = ForgeryDataset(args.data_dir, phase='val')
    solution, submission = [], []
    
    # Counters for analysis
    stats = {"auth_correct": 0, "auth_wrong": 0, "forged_caught": 0, "forged_missed": 0}

    print(f"📊 Processing {len(val_ds)} images...")
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            img, mask = val_ds[i]
            gt_mask = mask.squeeze().numpy().astype(np.uint8)
            gt_str = mask_to_kaggle_format(gt_mask)
            is_gt_forged = gt_mask.max() > 0
            
            input_t = img.unsqueeze(0).to(device)
            
            # --- STEP 1: CLASSIFICATION (SegFormer B2) ---
            out_cls = model_cls(input_t)
            if isinstance(out_cls, list): out_cls = out_cls[0]
            # Use max probability as the "Forgery Score"
            prob_cls_map = torch.sigmoid(out_cls).squeeze().cpu().numpy()
            forgery_score = prob_cls_map.max()
            
            # DECISION: Is it forged?
            is_predicted_forged = forgery_score > 0.5
            
            # --- STEP 2: SEGMENTATION (U-Net Baseline) ---
            if is_predicted_forged:
                # Run the Specialist U-Net
                out_seg = model_seg(input_t)
                if isinstance(out_seg, list): out_seg = out_seg[0]
                pred_bin = (torch.sigmoid(out_seg).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
                
                # Update Stats
                if is_gt_forged: stats["forged_caught"] += 1
                else: stats["auth_wrong"] += 1
                
            else:
                # Predict Authentic (Empty Mask)
                pred_bin = np.zeros_like(gt_mask)
                
                # Update Stats
                if not is_gt_forged: stats["auth_correct"] += 1
                else: stats["forged_missed"] += 1

            # Format for Kaggle Metric
            pred_str = mask_to_kaggle_format(pred_bin)
            solution.append({'row_id': i, 'annotation': gt_str, 'shape': json.dumps(mask.shape[1:])})
            submission.append({'row_id': i, 'annotation': pred_str})

    # 4. Compute Final Score
    score = kaggle_metric.score(pd.DataFrame(solution), pd.DataFrame(submission), 'row_id')
    
    print("\n" + "="*40)
    print(f"🏆 TWO-STAGE PIPELINE RESULTS")
    print(f"   🔹 oF1 Score: {score:.4f}")
    print("-" * 40)
    print("   🔍 Pipeline Behavior:")
    print(f"      ✅ Correctly ignored Authentic: {stats['auth_correct']}")
    print(f"      ❌ False Alarms (Authentic->Forged): {stats['auth_wrong']}")
    print(f"      ✅ Correctly flagged Forged: {stats['forged_caught']}")
    print(f"      ❌ Missed Forgeries: {stats['forged_missed']}")
    print("="*40)
    
    # Save results
    with open("pipeline_results.txt", "w") as f:
        f.write(f"oF1: {score}\nStats: {stats}")

if __name__ == "__main__":
    evaluate_pipeline()