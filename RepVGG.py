import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.nn.init as init
from torch import Tensor

#TBD
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class RepVGGBlock(nn.Module):

    def __init__(self, in_filters, out_filters):
        super(RepVGGBlock, self).__init__()


    def forward(self,x):
        if self.training:
            pass 
        else:
            return self.forward_inference(x)

    def reparam(self):
        pass 

    def forward_inference(self,x):
        pass
