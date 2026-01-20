import torch

def calculate_metrics(pred_tensor, target_tensor, threshold=0.5):
    """
    Calculates Dice, IoU, Precision, and Recall for binary tensors.
    """
    # Convert logits to probabilities
    pred_prob = torch.sigmoid(pred_tensor)
    pred_bin = (pred_prob > threshold).float()
    
    # Flatten tensors
    pred_flat = pred_bin.view(-1)
    target_flat = target_tensor.view(-1)
    
    # Constants for stability
    SMOOTH = 1e-6
    
    # True Positives, False Positives, False Negatives
    tp = (pred_flat * target_flat).sum()
    fp = pred_flat.sum() - tp
    fn = target_flat.sum() - tp
    
    # Metrics
    precision = (tp + SMOOTH) / (tp + fp + SMOOTH)
    recall = (tp + SMOOTH) / (tp + fn + SMOOTH)
    dice = (2 * tp + SMOOTH) / (2 * tp + fp + fn + SMOOTH)
    iou = (tp + SMOOTH) / (tp + fp + fn + SMOOTH)
    
    return {
        "Dice": dice.item(),
        "IoU": iou.item(),
        "Precision": precision.item(),
        "Recall": recall.item()
    }