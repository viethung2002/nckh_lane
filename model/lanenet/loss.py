# coding: utf-8
"""
This is the implementation of various loss functions for lane detection.
"""

from torch.nn.modules.loss import _Loss
from torch.autograd import Variable
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FocalLoss(nn.Module):
    '''
    Only consider two class now: foreground, background.
    '''
    def __init__(self, gamma=2, alpha=[0.5, 0.5], n_class=2, reduction='mean', device = DEVICE):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction
        self.n_class = n_class
        self.device = device

    def forward(self, input, target):
        pt = F.softmax(input, dim=1)
        pt = pt.clamp(min=0.000001,max=0.999999)
        target_onehot = torch.zeros((target.size(0), self.n_class, target.size(1),target.size(2))).to(self.device)
        loss = 0
        for i in range(self.n_class):
            target_onehot[:,i,...][target == i] = 1
        for i in range(self.n_class):
            loss -= self.alpha[i] * (1 - pt[:,i,...]) ** self.gamma * target_onehot[:,i,...] * torch.log(pt[:,i,...])

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
            
        return loss

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        if input.size(1) == 1:
            input = torch.sigmoid(input)
        else:
            input = F.softmax(input, dim=1)

        num_classes = input.size(1)
        target_onehot = torch.zeros_like(input)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(input * target_onehot, dims)
        cardinality = torch.sum(input + target_onehot, dims)
        dice_coeff = (2. * intersection + self.smooth) / (cardinality + self.smooth)

        dice_loss = 1 - dice_coeff

        if self.reduction == 'mean':
            dice_loss = dice_loss.mean()
        elif self.reduction == 'sum':
            dice_loss = dice_loss.sum()

        return dice_loss

class IoULoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean'):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        if input.size(1) == 1:
            input = torch.sigmoid(input)
        else:
            input = F.softmax(input, dim=1)

        num_classes = input.size(1)
        target_onehot = torch.zeros_like(input)
        target_onehot.scatter_(1, target.unsqueeze(1), 1)

        dims = (0,) + tuple(range(2, target.ndimension()))
        intersection = torch.sum(input * target_onehot, dims)
        union = torch.sum(input + target_onehot, dims) - intersection
        iou = (intersection + self.smooth) / (union + self.smooth)

        iou_loss = 1 - iou

        if self.reduction == 'mean':
            iou_loss = iou_loss.mean()
        elif self.reduction == 'sum':
            iou_loss = iou_loss.sum()

        return iou_loss

class CombinedLoss(nn.Module):
    def __init__(self, weights=[1, 1], reduction='mean'):
        super(CombinedLoss, self).__init__()
        self.weights = weights
        self.reduction = reduction
        self.ce_loss = nn.CrossEntropyLoss(reduction=reduction)
        self.dice_loss = DiceLoss(reduction=reduction)

    def forward(self, input, target):
        ce = self.ce_loss(input, target)
        dice = self.dice_loss(input, target)
        loss = self.weights[0] * ce + self.weights[1] * dice
        return loss

class SmoothL1Loss(nn.Module):
    def __init__(self, reduction='mean'):
        super(SmoothL1Loss, self).__init__()
        self.reduction = reduction
        self.loss_fn = nn.SmoothL1Loss(reduction=reduction)

    def forward(self, input, target):
        return self.loss_fn(input, target)

class DiscriminativeLoss(_Loss):
    def __init__(self, delta_var=0.5, delta_dist=1.5, norm=2, alpha=1.0, beta=1.0, gamma=0.001,
                 usegpu=False, size_average=True):
        super(DiscriminativeLoss, self).__init__(reduction='mean')
        self.delta_var = delta_var
        self.delta_dist = delta_dist
        self.norm = norm
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.usegpu = usegpu
        assert self.norm in [1, 2]

    def forward(self, input, target):

        return self._discriminative_loss(input, target)

    def _discriminative_loss(self, embedding, seg_gt):
        batch_size, embed_dim, H, W = embedding.shape
        embedding = embedding.reshape(batch_size, embed_dim, H*W)
        seg_gt = seg_gt.reshape(batch_size, H*W)

        var_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        dist_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)
        reg_loss = torch.tensor(0, dtype=embedding.dtype, device=embedding.device)

        for b in range(batch_size):
            embedding_b = embedding[b]  # (embed_dim, H*W)
            seg_gt_b = seg_gt[b]  # (H*W)

            labels, indexs = torch.unique(seg_gt_b, return_inverse=True)
            num_lanes = len(labels)
            if num_lanes == 0:
                _nonsense = embedding.sum()
                _zero = torch.zeros_like(_nonsense)
                var_loss = var_loss + _nonsense * _zero
                dist_loss = dist_loss + _nonsense * _zero
                reg_loss = reg_loss + _nonsense * _zero
                continue

            centroid_mean = []
            for lane_idx in labels:
                seg_mask_i = (seg_gt_b == lane_idx)

                if not seg_mask_i.any():
                    continue
                
                embedding_i = embedding_b * seg_mask_i
                mean_i = torch.sum(embedding_i, dim=1) / torch.sum(seg_mask_i)
                centroid_mean.append(mean_i)
                # ---------- var_loss -------------
                var_loss = var_loss + torch.sum(F.relu(
                    torch.norm(embedding_i[:,seg_mask_i] - mean_i.reshape(embed_dim, 1), dim=0) - self.delta_var) ** 2) / torch.sum(seg_mask_i) / num_lanes
            centroid_mean = torch.stack(centroid_mean)  # (n_lane, embed_dim)

            if num_lanes > 1:
                centroid_mean1 = centroid_mean.reshape(-1, 1, embed_dim)
                centroid_mean2 = centroid_mean.reshape(1, -1, embed_dim)

                dist = torch.norm(centroid_mean1 - centroid_mean2, dim=2)   # shape (num_lanes, num_lanes)
                dist = dist + torch.eye(num_lanes, dtype=dist.dtype,
                                        device=dist.device) * self.delta_dist

                # divided by two for double calculated loss above, for implementation convenience
                dist_loss = dist_loss + torch.sum(F.relu(-dist + self.delta_dist) ** 2) / (
                        num_lanes * (num_lanes - 1)) / 2

            # reg_loss is not used in original paper
            # reg_loss = reg_loss + torch.mean(torch.norm(centroid_mean, dim=1))

        var_loss = var_loss / batch_size
        dist_loss = dist_loss / batch_size
        reg_loss = reg_loss / batch_size

        return var_loss, dist_loss, reg_loss
    
def compute_loss(net_output, binary_label, instance_label, loss_type='FocalLoss'):
    """ Tính toán loss dựa trên đầu ra của mạng, nhãn nhị phân và nhãn instance. """

    k_binary = 10.0  # Hệ số cho binary loss
    k_instance = 0.3  # Hệ số cho instance loss
    k_dist = 1.0  # Hệ số cho loss phân biệt

    # Chọn hàm loss cho binary segmentation
    if loss_type == 'FocalLoss':
        loss_fn = FocalLoss(gamma=2, alpha=[0.25, 0.75])
    elif loss_type == 'CrossEntropyLoss':
        loss_fn = nn.CrossEntropyLoss()
    elif loss_type == 'DiceLoss':
        loss_fn = DiceLoss()
    elif loss_type == 'IoULoss':
        loss_fn = IoULoss()
    elif loss_type == 'CombinedLoss':
        loss_fn = CombinedLoss(weights=[1, 1])
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

    # Tính binary segmentation loss
    binary_seg_logits = net_output["binary_seg_logits"]
    binary_loss = loss_fn(binary_seg_logits, binary_label)

    # Tính instance segmentation loss
    pix_embedding = net_output["instance_seg_logits"]
    ds_loss_fn = DiscriminativeLoss(delta_var=0.5, delta_dist=1.5, norm=2, alpha=1.0, beta=1.0, gamma=0.001)
    var_loss, dist_loss, reg_loss = ds_loss_fn(pix_embedding, instance_label)

    # Nhân các hệ số trọng số cho từng loại loss
    binary_loss = binary_loss * k_binary
    var_loss = var_loss * k_instance
    dist_loss = dist_loss * k_dist

    instance_loss = var_loss + dist_loss
    total_loss = binary_loss + instance_loss

    out = net_output["binary_seg_pred"]  # Kết quả phân đoạn nhị phân

    return total_loss, binary_loss, instance_loss, out
