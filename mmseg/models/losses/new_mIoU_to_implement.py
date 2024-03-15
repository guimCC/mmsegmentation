import torch

class HistogramIoUMetric:
    def __init__(self, num_classes, ignore_index=None):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.reset()

    def reset(self):
        self.total_area_intersect = torch.zeros(self.num_classes)
        self.total_area_union = torch.zeros(self.num_classes)
        self.total_area_pred = torch.zeros(self.num_classes)
        self.total_area_label = torch.zeros(self.num_classes)

    def add_batch(self, pred_label, label):
        assert pred_label.shape == label.shape, "Prediction and label must have the same shape"

        if self.ignore_index is not None:
            mask = label != self.ignore_index
            pred_label = pred_label[mask]
            label = label[mask]

        intersect = pred_label[pred_label == label]
        area_intersect = torch.histc(intersect.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        area_pred_label = torch.histc(pred_label.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        area_label = torch.histc(label.float(), bins=self.num_classes, min=0, max=self.num_classes - 1)
        area_union = area_pred_label + area_label - area_intersect

        self.total_area_intersect += area_intersect
        self.total_area_union += area_union
        self.total_area_pred += area_pred_label
        self.total_area_label += area_label

    def compute_metrics(self):
        iou = self.total_area_intersect / (self.total_area_union + 1e-10)
        mean_iou = iou.mean()
        return {
            "IoU": iou,
            "mIoU": mean_iou
        }

# Example usage:
num_classes = 3  # Example number of classes
ignore_index = 255  # For pixels to be ignored

metric = HistogramIoUMetric(num_classes=num_classes, ignore_index=ignore_index)

# Assume pred_label and label are PyTorch tensors of shape (H, W) with class indices as values
# pred_label, label = ...

# Add batch of predictions and labels
metric.add_batch(pred_label, label)

# Compute and print metrics
results = metric.compute_metrics()
print(f"Mean IoU: {results['mIoU']:.4f}")
for i, iou in enumerate(results['IoU']):
    print(f"IoU for class {i}: {iou:.4f}")
