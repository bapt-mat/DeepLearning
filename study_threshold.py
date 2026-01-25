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

# Define thresholds to test
THRESHOLDS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

def mask_to_rle(binary_mask):
    labeled_mask, num_features = label(binary_mask)
    instances = []
    for k in range(1, num_features + 1):
        instances.append((labeled_mask == k).astype(np.uint8))
    return "authentic" if not instances else kaggle_metric.rle_encode(instances)

def run_study():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True) 
    parser.add_argument('--arch', type=str, default='segformer')
    parser.add_argument('--encoder', type=str, default='mit_b2')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Starting Threshold Study on {args.model_path}")
    
    # loading model
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None).to(device)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    val_ds = ForgeryDataset(args.data_dir, phase='val')
    
    # calculate predictions
    print("Generating probability maps...")
    ground_truth = []
    predictions = []
    
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            img, mask = val_ds[i]
            # Save GT
            gt_mask = mask.squeeze().numpy().astype(np.uint8)
            gt_rle = mask_to_rle(gt_mask)
            ground_truth.append({'row_id': i, 'annotation': gt_rle, 'shape': json.dumps(mask.shape[1:])})
            
            # Save Pred
            input_t = img.unsqueeze(0).to(device)
            out = model(input_t)
            if isinstance(out, list): out = out[0]
            prob_map = torch.sigmoid(out).squeeze().cpu().numpy()
            predictions.append(prob_map)

    sol_df = pd.DataFrame(ground_truth)
    scores = []

    print("Testing thresholds...")
    for t in THRESHOLDS:
        # Apply threshold 't' to all stored predictions
        submission = []
        for i, prob in enumerate(predictions):
            pred_bin = (prob > t).astype(np.uint8)
            pred_rle = mask_to_rle(pred_bin)
            submission.append({'row_id': i, 'annotation': pred_rle})
        
        sub_df = pd.DataFrame(submission)
        score = kaggle_metric.score(sol_df, sub_df, 'row_id')
        scores.append(score)
        print(f"Threshold {t:.1f}: oF1 = {score:.4f}")

    # Plotting
    plt.figure(figsize=(8, 6))
    plt.plot(THRESHOLDS, scores, marker='o', linestyle='-', color='b')
    plt.title(f"Threshold Sensitivity: {args.model_path}")
    plt.xlabel("Binarization Threshold")
    plt.ylabel("oF1 Score")
    plt.grid(True)
    plt.savefig(f"threshold_study.png")
    print("Plot saved to threshold_study.png")

if __name__ == "__main__":
    run_study()