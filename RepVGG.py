from pickle import DUP
import torch
from torch.nn.modules import padding
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.nn.init as init
from torch import Tensor
from utils import reparam_funcs
from utils.reparam_funcs import reparam_func
# Kaiming init doesn't play well with the weight reparam. To be fair, the paper doesn't use this init
def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        # init.kaiming_normal_(m.weight)
        # init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")

        # m.weight.data.normal_(0.0, 0.02)
        # init.normal_(m.weight,0,0.02)
        # init.xavier_normal_(m.weight)
        pass




class RepVGGBlock(nn.Module):
    """Single RepVGG block. We build these into distinct 'stages'"""

    def __init__(self, num_channels):
        super(RepVGGBlock, self).__init__()

        self.conv_3 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            bias=False,
        )
        self.bn_3 = nn.BatchNorm2d(num_features=num_channels)

        self.conv_1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(1, 1),
            padding=0,
            stride=1,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(num_features=num_channels)


        self.bn_0 = nn.BatchNorm2d(num_features=num_channels)


        # Reparam conv block
        self.rep_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            bias=True,
        )

        self.activation = nn.ReLU()
        self.reparam = False

    def forward(self, x):
        if not self.reparam:
            x_3 = self.bn_3(self.conv_3(x))
            x_1 = self.bn_1(self.conv_1(x))

            
            x_0 = self.bn_0(x)

            return self.activation(x_3 + x_1 + x_0)

        else:

            return self.activation(self.rep_conv(x))


class DownsampleRepVGGBlock(nn.Module):
    """Downsample RepVGG block. Comes at the end of a stage"""

    def __init__(self, in_channels, out_channels, stride = 2):
        super(DownsampleRepVGGBlock, self).__init__()

        self.conv_3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=stride,
            bias=False,
        )
        self.bn_3 = nn.BatchNorm2d(num_features=out_channels)

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=0,
            stride=stride,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(num_features=out_channels)

        # Reparam conv block
        self.rep_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=stride,
            bias=True,
        )
        self.reparam = False
        self.activation = nn.ReLU()

    def forward(self, x):
        if not self.reparam:
            x_3 = self.bn_3(self.conv_3(x))
            x_1 = self.bn_1(self.conv_1(x))
            
            return self.activation(x_3 + x_1)
        else:

            return self.activation(self.rep_conv(x))


class RepVGGStage(nn.Module):
    """Single RepVGG stage. These are stacked together to form the full RepVGG architecture"""

    def __init__(self, in_channels, out_channels, N, a, b=None, stride = 2):
        super(RepVGGStage, self).__init__()

        self.in_channels = in_channels if in_channels == 3 else int(a * in_channels)

        out_channels_ = min(out_channels, int(out_channels*a)) if in_channels == 3 else int(a * out_channels)

        self.out_channels = out_channels_ if b is None else int(b * out_channels)
        

        self.sequential = nn.Sequential(
            *[
                RepVGGBlock(num_channels=self.in_channels)
                for _ in range(0, N - 1)
            ]
            + [DownsampleRepVGGBlock(in_channels = self.in_channels, out_channels = self.out_channels, stride = stride)]
        )

        self.apply(_weights_init)

    def forward(self, x):
        return self.sequential(x)

    def _reparam(self):
        with torch.no_grad():
            for stage in self.sequential:
                reparam_weight, reparam_bias = reparam_func(stage)
                stage.rep_conv.weight.data = reparam_weight
                stage.rep_conv.bias.data = reparam_bias
                stage.reparam = True


    def switch_to_deploy(self):
        for stage in self.sequential:
            stage.reparam = True
            #delete old attributes
            if hasattr(stage, 'conv_3'):
                delattr(stage,'conv_3')
            
            if hasattr(stage, 'conv_1'):
                delattr(stage,'conv_1')

            if hasattr(stage, 'bn_1'):
                delattr(stage,'bn_1')
            
            if hasattr(stage, 'bn_0'):
                delattr(stage,'bn_0')
            
            if hasattr(stage, 'bn_3'):
                delattr(stage,'bn_3')

    def _train(self):
        with torch.no_grad():
            for stage in self.sequential:
                stage.reparam = False


class RepVGG(nn.Module):
    def __init__(
        self,
        filter_depth=[1, 2, 4, 14,1],
        filter_list=[64, 64, 128, 256, 512],
        stride=[1, 1, 2, 2, 2],
        a=1,
        b=2.5,
    ):
        super(RepVGG, self).__init__()

        self.stages = nn.Sequential(
            *[
                RepVGGStage(
                    in_channels=3,
                    out_channels=filter_list[0],
                    N=filter_depth[0],
                    a=a,
                    stride = stride[0]
                )
            ]
            + [
                RepVGGStage(
                    in_channels=filter_list[i - 1],
                    out_channels=filter_list[i],
                    N=filter_depth[i],
                    a=a,
                    stride = stride[i]
                )
                for i in range(1, len(filter_depth) - 1)
            ]
            + [
                RepVGGStage(
                    in_channels=filter_list[-2],
                    out_channels=filter_list[-1],
                    N=filter_depth[-1],
                    a=a,
                    b=b,
                    stride = stride[-1]
                )
            ]
        )

        self.fc = nn.Linear(in_features=int(b * filter_list[-1]), out_features=10)

        self.apply(_weights_init)

    def forward(self, x):

        x = self.stages(x)

        # Global average pooling
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)

        return self.fc(x)

    def _reparam(self):
        for stage in self.stages:
            stage._reparam()
    
    def _switch_to_deploy(self):
        for module in self.modules():
            if hasattr(module, 'switch_to_deploy'):
                module.switch_to_deploy()
        

    def _train(self):
        for stage in self.stages:
            stage._train()


def deploy_model(model):
    #Create a copy of model and switch to deploy
    deployed_model = copy.deepcopy(model)
    deployed_model._switch_to_deploy()
    return deployed_model




if __name__ == "__main__":

    model = RepVGG(
        filter_depth=[1, 2, 4, 14],
        filter_list=[16, 16, 32, 64],
        stride=[1, 1, 2, 2],
        a=0.75,
        b=2.5,
    )

    # QA
    model.eval()

    input = torch.randn(1,3,32,32)

    out_train = model(input)

    model._reparam()

    deployed_model = deploy_model(model)

    out_eval = deployed_model(input)

    print(((out_train - out_eval) ** 2).sum())



    
