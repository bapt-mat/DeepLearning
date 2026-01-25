import torch
import numpy as np
import pandas as pd
import json
import argparse
from scipy.ndimage import label
from tqdm import tqdm
from model import FlexibleModel
from dataset import ForgeryDataset
import kaggle_metric

def mask_to_kaggle_format(binary_mask):
    labeled_mask, num_features = label(binary_mask)
    instances = []
    for i in range(1, num_features + 1):
        instance = (labeled_mask == i).astype(np.uint8)
        if instance.sum() > 10: instances.append(instance)
    return "authentic" if not instances else kaggle_metric.rle_encode(instances)

def run_evaluation():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--arch', type=str, default='unet')
    parser.add_argument('--encoder', type=str, default='resnet34')
    parser.add_argument('--save_name', type=str, default='model')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load Model
    model = FlexibleModel(arch=args.arch, encoder=args.encoder, weights=None).to(device)
    model.load_state_dict(torch.load(f"{args.save_name}.pth", map_location=device))
    model.eval()

    val_ds = ForgeryDataset(args.data_dir, phase='val')
    solution, submission = [], []

    print(f"Evaluating {args.save_name} on {len(val_ds)} images...")
    with torch.no_grad():
        for i in tqdm(range(len(val_ds))):
            img, mask = val_ds[i]
            gt_str = mask_to_kaggle_format(mask.squeeze().numpy().astype(np.uint8))
            
            # Predict
            out = model(img.unsqueeze(0).to(device))
            if isinstance(out, list): out = out[0] # Handle DeepSup
            pred_bin = (torch.sigmoid(out).squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            pred_str = mask_to_kaggle_format(pred_bin)
            
            solution.append({'row_id': i, 'annotation': gt_str, 'shape': json.dumps(mask.shape[1:])})
            submission.append({'row_id': i, 'annotation': pred_str})

    score = kaggle_metric.score(pd.DataFrame(solution), pd.DataFrame(submission), 'row_id')
    print(f"oF1 SCORE: {score:.5f}")
    
    with open(f"score_{args.save_name}.txt", "w") as f:
        f.write(str(score))

if __name__ == "__main__":
    run_evaluation()