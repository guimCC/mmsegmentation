import torch.nn as nn

from mmseg.registry import MODELS

# Decorator to register mobilenet as a backbone model
@MODELS.register_module()
class MobileNet(nn.Module):
    
    def __init__(self, arg1, arg2):
        # ...
        pass
    
    def forward(self, x):
        # ...
        pass
    
    def init_weights(self, pretrained=None):
        # ...
        pass

# must import at mmseg/models/backbones/__init__.py as from .mobilenet import MobileNet