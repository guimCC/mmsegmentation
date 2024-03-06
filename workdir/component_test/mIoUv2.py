import torch
import numpy as np

class IoU():
    def __init__(self, num_classes, ignore_index = -1):
        self.num_classes = num_classes
        self.ignore_index = ignore_index

    
    def batch_intersection_union(self, pred, target):
        results = []
        for i in range(pred.shape[0]):
            results.append(self.intersect_and_union(pred[i], target[i]))
        return results
    
    def intersect_and_union(self, pred_label: torch.tensor, label: torch.tensor):
        """Calculate Intersection and Union.

        Args:
            pred_label (torch.tensor): Prediction segmentation map
                or predict result filename. The shape is (H, W).
            label (torch.tensor): Ground truth segmentation map
                or label filename. The shape is (H, W).
            num_classes (int): Number of categories.
            ignore_index (int): Index that will be ignored in evaluation.

        Returns:
            torch.Tensor: The intersection of prediction and ground truth
                histogram on all classes.
            torch.Tensor: The union of prediction and ground truth histogram on
                all classes.
            torch.Tensor: The prediction histogram on all classes.
            torch.Tensor: The ground truth histogram on all classes.
        """

        mask = (label != self.ignore_index)
        pred_label = pred_label[mask]
        label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(
            intersect.float(), bins=(self.num_classes), min=0,
            max=self.num_classes - 1).cpu()
        area_pred_label = torch.histc(
            pred_label.float(), bins=(self.num_classes), min=0,
            max=self.num_classes - 1).cpu()
        area_label = torch.histc(
            label.float(), bins=(self.num_classes), min=0,
            max=self.num_classes - 1).cpu()
        area_union = area_pred_label + area_label - area_intersect
        return area_intersect, area_union, area_pred_label, area_label
    
    def compute_total(self, results):
        results = tuple(zip(*results))
        #print(len(results))
        assert len(results) == 4

        total_area_intersect = sum(results[0])
        total_area_union = sum(results[1])
        total_area_pred_label = sum(results[2])
        total_area_label = sum(results[3])
        
        return total_area_intersect, total_area_union, total_area_pred_label, total_area_label
    
    def iou(self, total_area_intersect, total_area_union, total_area_pred_label, total_area_label):
        """Calculate Intersection over Union.

        Args:
            total_area_intersect (torch.Tensor): The intersection of prediction
                and ground truth histogram on all classes.
            total_area_union (torch.Tensor): The union of prediction and ground
                truth histogram on all classes.

        Returns:
            torch.Tensor: Intersection over Union.
        """
        iou = total_area_intersect / total_area_union#(total_area_union + 1e-10)
        print(iou)
        
        return np.nanmean(iou)
    
    def compute_reward(self, pred_label, label):
        pred = torch.argmax(pred_label, dim=1)
        results = self.batch_intersection_union(pred, label)

        a, b, c, d = self.compute_total(results)

        return self.iou(a, b, c, d)
