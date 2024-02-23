import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.registry import MODELS
#from .utils import weight_loss



def RewardWeightedLogLikelihoodLoss(pred, target, reward, ignore_index=255):
    #assert pred.size() == target.size() and target.numel() > 0
    #print(pred.size(), target.size(), reward.size())
    
    
    
    # we are only interested in the log_probabilities of the target classes
    # log probs is a tensor of shape (batch_size, num_classes) containing
    # the log probabilities of each class for each pixel
    # then, by using gather, we only take those logits from the true class
    # for each pixel
    
    log_probs = F.log_softmax(pred, dim=1)
    target_masked = torch.where(target == ignore_index, torch.zeros_like(target), target)
    log_probs_target = torch.gather(log_probs, 1, target_masked.unsqueeze(1)).squeeze(1)
    
    valid_mask = (target != ignore_index).float()
    weighted_log_probs = reward * log_probs_target * valid_mask
    valid_pixels = valid_mask.sum()
    
    if valid_pixels > 0:
        loss = -weighted_log_probs.sum() / valid_pixels
    else:
        loss = torch.tensor(0.0, requires_grad=True, device=pred.device)
    #print(log_probs[0])

    #print("Unique target values:", torch.unique(target))
    #print("Min value in target:", torch.min(target))
    #print("Max value in target:", torch.max(target))

    #not_ignore_mask = (target != ignore_index)
    #target = target * not_ignore_mask.long()

    #log_probs_target = torch.gather(log_probs, 1, target.unsqueeze(1)).squeeze(1)
    
    #weighted_log_probs = reward * log_probs_target * not_ignore_mask

    #loss = -weighted_log_probs[not_ignore_mask].mean()    
    #print("SAMBIÑAAAAAAAAAAAAAAAAAAAAAAAA")
    print("AAAAAAAAAAAAAAAAA->", loss)
    print("BBBBBBBBBBBBBBBBB->", reward)
    
    return loss #torch.tensor(0.5, requires_grad=True)

@MODELS.register_module()
class ReinforceLoss(nn.Module):
    def __init__(self, reduction='mean', loss_weight=1.0, loss_name= 'loss_rein'):
        super(ReinforceLoss, self).__init__()
        self.reduction = reduction
        self.loss_weight = loss_weight
        self._loss_name = loss_name # must add for backpropagation

    def forward(self, pred, target, weight=None, avg_factor=None, reduction_override=None, ignore_index=-100, reward=None):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss = self.loss_weight * RewardWeightedLogLikelihoodLoss(
            pred, target, reward)#weight, reduction=reduction, avg_factor=avg_factor)
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