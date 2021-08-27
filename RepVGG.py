import torch
from torch.nn.modules import padding
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
    """Single RepVGG block. We build these into distinct 'stages'

    """

    def __init__(self, in_channels, out_channels):
        super(RepVGGBlock, self).__init__()

        self.conv_3 = nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size=(3,3), padding = 1, stride = 1, bias=False)
        self.bn_3 = nn.BatchNorm2d(num_features=out_channels)

        self.conv_1 = nn.Conv2d(in_channels=in_channels, out_channels= out_channels, kernel_size=(1,1), padding = 0, stride = 1)
        self.bn_1 = nn.BatchNorm2d(num_features=out_channels)


        self.identity = False
        if in_channels == out_channels:
            #Use identity branch
            self.bn_0 = nn.BatchNorm2d(num_features=out_channels)
            self.identity = True



    def forward(self,x):
        x_3 = self.bn_3(self.conv_3(x))
        x_1 = self.bn_1(self.conv_1(x))

        if self.identity:
            x_0 = self.bn_0(x)

            return F.relu(x_3+x_1+x_0)

        else:
            return F.relu(x_3+x_1)

    def reparam(self):
        pass 

    def forward_inference(self,x):
        pass


class DownsampleRepVGGBlock(nn.Module):
    """Single RepVGG block. We build these into distinct 'stages'

    """

    def __init__(self, num_channels):
        super(DownsampleRepVGGBlock, self).__init__()

        self.conv_3 = nn.Conv2d(in_channels= num_channels, out_channels=  num_channels, kernel_size=(3,3), padding = 1, stride = (2,2), bias=False)
        self.bn_3 = nn.BatchNorm2d(num_features= num_channels)

        self.conv_1 = nn.Conv2d(in_channels= num_channels, out_channels=  num_channels, kernel_size=(1,1), padding = 0, stride = (2,2))
        self.bn_1 = nn.BatchNorm2d(num_features= num_channels)

        #We pad the identity the usual way
        self.bn_0 = nn.BatchNorm2d(num_features= num_channels)

    def forward(self,x):
        x_3 = self.bn_3(self.conv_3(x))
        x_1 = self.bn_1(self.conv_1(x))

        x_0 = self.bn_0(x[:, :, ::2, ::2])

        return F.relu(x_3+x_1+x_0)



    def reparam(self):
        pass 

    def forward_inference(self,x):
        pass


class RepVGGStage(nn.Module):
    """Single RepVGG stage. These are stacked together to form a full RepVGG model

    """
    def __init__(self,in_channels,out_channels, N, a, b):
        super(RepVGGStage, self).__init__()

        self.in_channels = a*in_channels
        self.out_channels = a*out_channels

        self.sequential = nn.Sequential(
            *[RepVGGBlock(in_channels = in_channels, out_channels=self.out_channels)]
            + [RepVGGBlock(in_channels = self.out_channels, out_channels=self.out_channels) for _ in range(0,N-2)]
            +[DownsampleRepVGGBlock(num_channels=self.out_channels)]
            )

    def forward(self,x):
        return self.sequential(x)


if __name__ == '__main__':

    model = nn.Sequential(RepVGGStage(in_channels=3, out_channels=64, N = 1, a = 1, b= 1),
                            RepVGGStage(in_channels=64, out_channels=64, N = 2, a = 1, b= 1),
                            RepVGGStage(in_channels=64, out_channels=128, N = 4, a = 1, b= 1),
                            RepVGGStage(in_channels=128, out_channels=256, N = 4, a = 1, b= 1),
                            )

    img = torch.ones(1,3,32,32)

    print(model(img).shape)


    


