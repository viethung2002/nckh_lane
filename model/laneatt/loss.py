# coding: utf-8
"""
This is the implementation of various loss functions for LaneATT lane detection.
"""

from torch.nn.modules.loss import _Loss
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.Module):
    '''
    Focal loss function to handle class imbalance, particularly useful in lane detection.
    '''
    def __init__(self, gamma=2, alpha=[0.5, 0.5], n_class=2, reduction='mean', device=DEVICE):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.n_class = n_class
        self.device = device

    def forward(self, input, target):
        pt = F.softmax(input, dim=1)
        pt = pt.clamp(min=0.000001, max=0.999999)
        target_onehot = torch.zeros((target.size(0), self.n_class, target.size(1), target.size(2))).to(self.device)
        loss = 0
        for i in range(self.n_class):
            target_onehot[:, i, ...][target == i] = 1
        for i in range(self.n_class):
            loss -= self.alpha[i] * (1 - pt[:, i, ...]) ** self.gamma * target_onehot[:, i, ...] * torch.log(pt[:, i, ...])

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        
        return loss

class SmoothL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_fn = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, input, target):
        return self.loss_fn(input, target)

class LaneATTLoss(nn.Module):
    def __init__(self, classification_weight=1.0, regression_weight=1.0, reduction='mean'):
        super(LaneATTLoss, self).__init__()
        self.classification_weight = classification_weight
        self.regression_weight = regression_weight
        self.focal_loss = FocalLoss(gamma=2, alpha=[0.25, 0.75], reduction=reduction)
        self.regression_loss = SmoothL1Loss(reduction=reduction)

    def forward(self, predictions, targets):
        # Predictions contain both classification and regression results
        class_logits, reg_output = predictions["class_logits"], predictions["reg_output"]
        class_labels, reg_targets = targets["class_labels"], targets["reg_targets"]

        # Classification loss (using Focal Loss to handle class imbalance)
        cls_loss = self.focal_loss(class_logits, class_labels)

        # Regression loss (using Smooth L1 Loss for position prediction)
        reg_loss = self.regression_loss(reg_output, reg_targets)

        # Combine both losses with respective weights
        total_loss = self.classification_weight * cls_loss + self.regression_weight * reg_loss
        
        return total_loss, cls_loss, reg_loss


def lossLaneatt(net_output, class_labels, reg_targets, loss_type='LaneATTLoss'):
    """
    Compute the loss based on the model outputs and targets for LaneATT.
    
    Args:
        net_output (dict): The output from the network containing classification and regression outputs.
        class_labels (Tensor): Ground truth labels for classification.
        reg_targets (Tensor): Ground truth regression targets for lane points.
        loss_type (str): The type of loss function to use.
    
    Returns:
        total_loss (Tensor): The computed total loss.
        cls_loss (Tensor): The classification loss.
        reg_loss (Tensor): The regression loss.
    """
    # Set up the loss function based on the loss type
    if loss_type == 'LaneATTLoss':
        loss_fn = LaneATTLoss(classification_weight=1.0, regression_weight=1.0)
    elif loss_type == 'SmoothL1Loss':
        loss_fn = SmoothL1Loss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")
    
    # Compute the combined loss
    total_loss, cls_loss, reg_loss = loss_fn(net_output, {"class_labels": class_labels, "reg_targets": reg_targets})
    
    return total_loss, cls_loss, reg_loss
