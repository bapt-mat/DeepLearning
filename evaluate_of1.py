import torch
import numpy as np
from scipy.ndimage import label
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm
from model import SimpleUNet
from dataset import ForgeryDataset

# --- UPDATE PATHS HERE ---
MODEL_PATH = "model_epoch_0.pth"  # Your saved model
DATA_PATH = "/Users/baptiste/MASTER/M2/S3/DeepLearning/dataset" # Local path
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
THRESHOLD = 0.5

def compute_pixel_f1(pred_mask, gt_mask):
    tp = np.sum(pred_mask * gt_mask)
    fp = np.sum(pred_mask) - tp
    fn = np.sum(gt_mask) - tp
    if tp == 0: return 0.0
    return (2 * tp) / (2 * tp + fp + fn)

def compute_oF1_single_image(pred_prob_map, gt_mask_map, threshold=0.5):
    pred_bin = (pred_prob_map > threshold).astype(np.uint8)
    gt_bin = (gt_mask_map > 0).astype(np.uint8)

    pred_labeled, num_pred = label(pred_bin, structure=np.ones((3,3)))
    gt_labeled, num_gt = label(gt_bin, structure=np.ones((3,3)))

    if num_pred == 0 and num_gt == 0: return 1.0
    if num_pred == 0 or num_gt == 0: return 0.0

    f1_matrix = np.zeros((num_gt, num_pred))
    for i in range(1, num_gt + 1):
        gt_instance = (gt_labeled == i)
        for j in range(1, num_pred + 1):
            pred_instance = (pred_labeled == j)
            f1_matrix[i-1, j-1] = compute_pixel_f1(pred_instance, gt_instance)

    row_ind, col_ind = linear_sum_assignment(f1_matrix, maximize=True)
    total_f1 = f1_matrix[row_ind, col_ind].sum()
    return total_f1 / max(num_gt, num_pred)

def evaluate_model():
    model = SimpleUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    val_ds = ForgeryDataset(DATA_PATH, phase='val')
    print(f"Evaluating oF1 on {len(val_ds)} images...")

    of1_scores = []
    with torch.no_grad():
        for img_tensor, mask_tensor in tqdm(val_ds):
            input_batch = img_tensor.unsqueeze(0).to(DEVICE)
            gt_mask_np = mask_tensor.squeeze().numpy()
            
            output = model(input_batch)
            pred_prob_np = torch.sigmoid(output).squeeze().cpu().numpy()
            
            score = compute_oF1_single_image(pred_prob_np, gt_mask_np, THRESHOLD)
            of1_scores.append(score)

    print(f"Global Validation oF1 Score: {np.mean(of1_scores):.4f}")

if __name__ == "__main__":
    evaluate_model()