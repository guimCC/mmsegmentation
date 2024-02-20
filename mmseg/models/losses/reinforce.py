import torch
import torch.nn as nn

from mmseg.registry import MODELS
from .utils import weight_loss


@weighted_loss
def RewardWeightedLogLikelihoodLoss(pred, target, reward):
    assert pred.size() == target.size() and target.numel() > 0
    
    log_probs = nn.log_softmax(pred)
    
    # we are only interested in the log_probabilities of the target classes
    # log probs is a tensor of shape (batch_size, num_classes) containing
    # the log probabilities of each class for each pixel
    # then, by using gather, we only take those logits from the true class
    # for each pixel
    log_probs_target = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)
    
    weighted_log_probs = reward * log_probs_target

    loss = -weighted_log_probs.mean()    
    
    return loss

@MODELS.register_module()
class ReinforceLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0):
        super(MyLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * RewardWeightedLogLikelihoodLoss(
            pred, target, weight, reduction=reduction, avg_factor=avg_factor)
        return loss