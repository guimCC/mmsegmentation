import warnings
from abc import ABCMeta, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn
from mmengine.model import BaseModule
from torch import Tensor

import numpy as np

from mmseg.structures import build_pixel_sampler
from mmseg.utils import ConfigType, SampleList
from ..builder import build_loss
from ..losses import accuracy
from ..utils import resize
from mmseg.registry import MODELS


class ReinforceDecodeHead(BaseDecodeHead):
    """
        Extends the Base Decode Head to implement the logif for the reward-based loss computation
    """
    def __init__()