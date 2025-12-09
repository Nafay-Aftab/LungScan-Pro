import torch

def calculate_metrics(pred, target, threshold=0.5):
    """
    Calculates Dice, IoU (Jaccard), Precision, and Recall.
    pred: (Batch, 1, H, W) raw logits or probabilities
    target: (Batch, 1, H, W) binary mask (0 or 1)
    """
    # Convert to binary
    pred = (pred > threshold).float()
    target = target.float()
    
    # Flatten
    pred = pred.view(-1)
    target = target.view(-1)
    
    # Components
    intersection = (pred * target).sum()
    total_pred_pixels = pred.sum()
    total_target_pixels = target.sum()
    
    # Dice Coefficient
    dice = (2.0 * intersection) / (total_pred_pixels + total_target_pixels + 1e-8)
    
    # Intersection over Union (IoU)
    # IoU = Intersection / (Area Pred + Area Target - Intersection)
    iou = intersection / (total_pred_pixels + total_target_pixels - intersection + 1e-8)
    
    # Precision (How many selected pixels were actually lung?)
    precision = intersection / (total_pred_pixels + 1e-8)
    
    # Recall (How many lung pixels did we find?)
    recall = intersection / (total_target_pixels + 1e-8)
    
    return dice.item(), iou.item(), precision.item(), recall.item()