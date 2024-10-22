import torch
import torch.nn as nn
import torch.nn.functional as F

class DiceLoss(nn.Module):
    def __init__(self, smooth=1, reduction='mean'):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.reduction = reduction

    def forward(self, input, target):
        # Chuyển đầu vào thành xác suất nếu cần
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

class BinaryCrossEntropyLoss(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropyLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss()

    def forward(self, input, target):
        return self.bce_loss(input, target)

class CSNNLoss(nn.Module):
    def __init__(self, loss_type='CrossEntropyLoss', smooth=1, reduction='mean'):
        super(CSNNLoss, self).__init__()
        if loss_type == 'CrossEntropyLoss':
            self.loss_fn = nn.CrossEntropyLoss()
        elif loss_type == 'DiceLoss':
            self.loss_fn = DiceLoss(smooth=smooth, reduction=reduction)
        elif loss_type == 'BCEWithLogitsLoss':
            self.loss_fn = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unsupported loss type: {loss_type}")

    def forward(self, inputs, targets):
        """
        Forward pass qua loss function.
        - inputs: đầu vào từ mô hình có kích thước [batch_size, num_classes, height, width].
        - targets: nhãn thực tế có kích thước [batch_size, height, width].
        """
        loss = self.loss_fn(inputs, targets)
        return loss
