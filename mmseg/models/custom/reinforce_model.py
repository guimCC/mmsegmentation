from mmengine.model import BaseModel
import torch.nn as nn
import torch


# Custom model definition. Using BaseModel not module since we are
# collapsing all the model's training logic into a single class
class ReinforceModel(BaseModel):
    def __init__(self, num_classes, **kwargs):
        
        super(ReinforceModel, self).__init__()
        self.num_classes = num_classes
        
        # define structure -> must align with pretrained model
        