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

class RepVGG(nn.Module):

    def __init__(self,filter_depth = [1,2,4,14,1],filter_list=[64, 64,128,256,512], a=1, b=1):
        super(RepVGG, self).__init__()

        self.stages = nn.Sequential(
            *[RepVGGStage(in_channels=3, out_channels=filter_list[0], N = filter_depth[0], a = a, b = b)]+
            [RepVGGStage(in_channels=filter_list[i-1], out_channels=filter_list[i], N = filter_depth[i], a = a, b = b) for i in range(1,5)
        ])

    
        self.fc = nn.Linear(in_features=filter_list[-1], out_features=10)
    

    def forward(self,x):

        x = self.stages(x)

        x = torch.mean(x,axis = (2,3))

        return self.fc(x)




if __name__ == '__main__':

    model = RepVGG()

    # img = torch.ones(1,3,112,112)
    model = RepVGGBlock(in_channels=64,out_channels=64)

    

    #Recipe is:

    # 1. Create 3 3x3 kernels and bias vectors
    
    reparam_weight = torch.zeros_like(model.conv_3.weight)

    #Size of out filter
    reparam_bias = torch.zeros(64)


    #3x3
    std = (model.bn_3.running_var + model.bn_3.eps).sqrt()
    t = (model.bn_3.weight / std).reshape(-1, 1, 1, 1)

    reparam_weight_3 = model.conv_3.weight*t
    reparam_bias_3 = -(model.bn_3.running_mean*model.bn_3.weight/model.bn_3.running_var) + model.bn_3.bias


    #1x1
    std = (model.bn_1.running_var + model.bn_1.eps).sqrt()
    t = (model.bn_1.weight / std).reshape(-1, 1, 1, 1)

    reparam_weight_1 = model.conv_1.weight*t
    reparam_bias_1 = -(model.bn_1.running_mean*model.bn_1.weight/model.bn_1.running_var) + model.bn_1.bias

    #idx
    std = (model.bn_0.running_var + model.bn_0.eps).sqrt()
    t = (model.bn_0.weight / std).reshape(-1, 1, 1, 1)

    reparam_weight_0 = torch.ones([64,64,1,1])*t
    reparam_bias_0 = -(model.bn_0.running_mean*model.bn_0.weight/model.bn_0.running_var) + model.bn_0.bias


    reparam_weight = reparam_weight_3
    reparam_bias = reparam_bias_3

    reparam_weight +=  F.pad(reparam_weight_1,(1,1,1,1),mode = 'constant',value=0)
    reparam_bias += reparam_bias_1

    reparam_weight +=  F.pad(reparam_weight_0,(1,1,1,1),mode = 'constant',value=0)
    reparam_bias += reparam_bias_0


    print(reparam_weight.shape)
    print(reparam_bias.shape)

    # # reparam_bias = None

    # reparam_weight += model.conv_3.weight
    # reparam_weight += F.pad(model.conv_1.weight,(1,1,1,1),mode = 'constant',value=0)

    
    # print(model.conv_3.weight.shape)
    

    # #Pad these 
    # # print(F.pad(model.conv_1.weight,(1,1,1,1),mode = 'constant',value=0).shape)




    


