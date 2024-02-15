import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .utils import weight_loss

@weighted_loss
def my_loss(pred, target):
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss

class MyLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * my_loss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss
    
# Must add at mmseg/models/losses/__init__.py -> from .test import MyLoss

# To use, modify loss_xxx field. Modify loss_decode field in head, loss_weight for multiple losses
# loss_decode = dict(type='MyLoss', loss_weight=1.0)