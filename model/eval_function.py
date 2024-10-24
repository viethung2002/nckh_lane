import torch
from torch.autograd import Variable
import numpy as np



# Function to calculate mAP
def calculate_map(binary_preds, binary_labels, thresholds=np.linspace(0.0, 1.0, 11)):
    """Calculate mean Average Precision (mAP) based on multiple thresholds."""
    precision_at_thresholds = []
    
    for threshold in thresholds:
        # Apply threshold
        binary_preds_thresholded = (binary_preds >= threshold).float()

        # Calculate true positives, false positives, false negatives
        tp = torch.sum((binary_preds_thresholded == 1) & (binary_labels == 1)).item()
        fp = torch.sum((binary_preds_thresholded == 1) & (binary_labels == 0)).item()
        fn = torch.sum((binary_preds_thresholded == 0) & (binary_labels == 1)).item()

        # Calculate precision and recall
        precision = tp / (tp + fp + 1e-6) if (tp + fp) != 0 else 0
        recall = tp / (tp + fn + 1e-6) if (tp + fn) != 0 else 0

        # Store precision value at this threshold
        precision_at_thresholds.append(precision)
    
    # Calculate mean Average Precision
    mean_ap = np.mean(precision_at_thresholds)
    return mean_ap

def dice_coeff(pred, target):
    smooth = 1.
    num = pred.size(0)
    m1 = pred.view(num, -1)  # Flatten
    m2 = target.view(num, -1)  # Flatten
    intersection = (m1 * m2).sum()
 
    return (2. * intersection + smooth) / (m1.sum() + m2.sum() + smooth)

class Eval_Score():
    # IoU and F1(Dice)

    def __init__(self, y_pred, y_true, threshold = 0.5):
        input_flatten = np.int32(y_pred.flatten() > threshold)
        target_flatten = np.int32(y_true.flatten() > threshold)
        self.intersection = np.sum(input_flatten * target_flatten)
        self.sum = np.sum(target_flatten) + np.sum(input_flatten)
        self.union = self.sum - self.intersection
    
    def Dice(self, eps=1):
        return np.clip(((2. * self.intersection) / (self.sum + eps)), 1e-5, 0.99999)
    
    def IoU(self):
        return self.intersection / self.union
    def F1(self):
        return 2 * self.intersection / (self.sum)
