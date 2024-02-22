import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
#from .utils import weight_loss


#@weight_loss
def IntersectionOverUnionLoss(pred, target, eps):
    # Assuming pred is of shape [N, C, H, W] and target is [N, H, W]
    # Convert predictions to class labels
    pred = F.softmax(pred, dim=1)
    pred = torch.argmax(pred, dim=1)
    
    # Ensure target is the same dtype as pred
    target = target.long()
    
    # Calculate IoU for each class and then average across classes
    ious = []
    for clss in range(pred.shape[1]):  # iterate over each class
        pred_inds = pred == clss
        target_inds = target == clss
        intersection = (pred_inds & target_inds).float().sum()
        union = (pred_inds | target_inds).float().sum() + eps
        ious.append((intersection / union).item())

    # Average IoU across classes
    mean_iou = sum(ious) / len(ious)
    
    # IoU loss
    loss = 1 - mean_iou
    
    return torch.tensor(loss, requires_grad=True)

@MODELS.register_module()
class IoULoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, eps = 1e-6, loss_name= 'loss_iou'):
        super(IoULoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.eps = eps
        self._loss_name = loss_name # must add fors backpropagation

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, ignore_index=-100):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * IntersectionOverUnionLoss(
            pred, target, self.eps)
        return loss
         
    @property
    def loss_name(self):
        """Loss Name.

        This function must be implemented and will return the name of this
        loss function. This name will be used to combine different loss items
        by simple sum operation. In addition, if you want this loss item to be
        included into the backward graph, `loss_` must be the prefix of the
        name.

        Returns:
            str: The name of this loss item.
        """
        return self._loss_name