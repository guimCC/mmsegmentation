import torch
import torch.nn as nn
import torch.nn.functional as F

def IntersectionOverUnionLoss(pred, target, eps):
    # Assuming pred is of shape [N, C, H, W] and target is [N, H, W]
    # Where N is the batch size, C is the number of classes, H is the height and W is the width
    # Convert predictions to class labels
    n_classes = pred.shape[1]
    pred = F.softmax(pred, dim=1)
    print("Softmax:", pred)
    pred = torch.argmax(pred, dim=1)
    print("Argmax:", pred)
    
    # Ensure target is the same dtype as pred
    target = target.long()
    print("Target:", target)
    
    # Calculate IoU for each class and then average across classes
    ious = []
    #print("Pred shape:", n_classes)
    for clss in range(n_classes):  # iterate over each class
        pred_inds = pred == clss # pixels predicted as class clss
        #print("Pred inds for class {}:".format(clss), pred_inds)
        
        target_inds = target == clss # pixels where the true class is clss
        #print("Target inds for class {}:".format(clss), target_inds)
        
        intersection = (pred_inds & target_inds).float().sum()
        #print("Intersection for class {}:".format(clss), intersection)
        
        union = (pred_inds | target_inds).float().sum() + eps
        #print("Union for class {}:".format(clss), union)
        
        if union.item() > eps:
            #print("IoU for class {}:".format(clss), (intersection / union).item())
            ious.append((intersection / union).item())

    # Average IoU across classes
    mean_iou = sum(ious) / len(ious)
    
    # IoU loss
    # loss = 1 - mean_iou
    loss = mean_iou
    
    return torch.tensor(loss, requires_grad=True, device=pred.device)