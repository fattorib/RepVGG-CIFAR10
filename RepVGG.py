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

    def __init__(self, in_channels, out_channels, stride=2):
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

    def __init__(self, in_channels, out_channels, N,stride=2):
        super(RepVGGStage, self).__init__()

        self.in_channels = in_channels 

        self.out_channels = out_channels

        self.sequential = nn.Sequential(
            *[
                DownsampleRepVGGBlock(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    stride=stride,
                )
            ]
            + [RepVGGBlock(num_channels=self.out_channels) for _ in range(0, N - 1)]
        )
        

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
            # delete old attributes
            if hasattr(stage, "conv_3"):
                delattr(stage, "conv_3")

            if hasattr(stage, "conv_1"):
                delattr(stage, "conv_1")

            if hasattr(stage, "bn_1"):
                delattr(stage, "bn_1")

            if hasattr(stage, "bn_0"):
                delattr(stage, "bn_0")

            if hasattr(stage, "bn_3"):
                delattr(stage, "bn_3")

    def _train(self):
        for stage in self.sequential:
            stage.reparam = False


class RepVGG(nn.Module):
    def __init__(
        self,
        filter_depth=[1, 2, 4, 14, 1],
        filter_list=[16, 16, 32, 32, 64],
        stride=[1, 1, 1, 2, 2],
        width_factor = [1,1,1,2.5],
    ):
        super(RepVGG, self).__init__()

        filter_list[0] = min(16,16*width_factor[0])

        # filter_list[1:] *= width_factor[1:]
        for i in range(1,len(filter_list)):
            filter_list[i] = int(filter_list[i]*width_factor[i])

        width_factor = [1,1,1,1]

        self.stages = nn.Sequential(
            *[
                RepVGGStage(
                    in_channels=3,
                    out_channels=int(filter_list[0]),
                    N=filter_depth[0],
                    stride=stride[0],
                )
            ]
            + [
                RepVGGStage(
                    in_channels=int(filter_list[i - 1]),
                    out_channels=filter_list[i],
                    N=filter_depth[i],
                    stride=stride[i],
                )
                for i in range(1, len(filter_depth)-1)
            ]
            + [
                RepVGGStage(
                    in_channels=int(filter_list[-2]),
                    out_channels=filter_list[-1],
                    N=filter_depth[-1],
                    stride=stride[-1],
                )
                
            ]
        )

        
    
        self.fc = nn.Linear(in_features=int(filter_list[-1]), out_features=10)

        self.gap = nn.AdaptiveAvgPool2d(output_size=1)

    def forward(self, x):

        x = self.stages(x)

        x = self.gap(x)
        x = x.view(x.size(0), -1)

        return self.fc(x)

    def _reparam(self):
        for stage in self.stages:
            stage._reparam()

    def _switch_to_deploy(self):
        for module in self.modules():
            if hasattr(module, "switch_to_deploy"):
                module.switch_to_deploy()

    def _train(self):
        for stage in self.stages:
            stage._train()


def deploy_model(model):
    # Create a copy of model and switch to deploy
    deployed_model = copy.deepcopy(model)

    deployed_model._reparam()

    deployed_model._switch_to_deploy()
    return deployed_model


def create_RepVGG_A0():
    return RepVGG(
        filter_depth=[1, 2, 4, 14, 1],
        filter_list=[16, 16, 32, 64, 64],
        stride=[1, 1, 2, 2, 1],
        width_factor=[0.75]*4 + [2.5]
    )


def create_RepVGG_A1():
    return RepVGG(
        filter_depth=[1, 2, 4, 14, 1],
        filter_list=[16, 16, 32, 64, 64],
        stride=[1, 1, 2, 2, 1],
        width_factor=[1]*4 + [2.5]
    )


def create_RepVGG_A2():
    return RepVGG(
        filter_depth=[1, 2, 4, 14, 1],
        filter_list=[16, 16, 32, 64, 64],
        stride=[1, 1, 2, 2, 1],
        width_factor=[1.5]*4 + [2.75]
    )


def create_RepVGG_B0():
    return RepVGG(
        filter_depth=[1, 4, 6, 16, 1],
        filter_list=[16, 16, 32, 64, 64],
        # filter_list=[16, 16, 32, 64, 128],
        stride=[1, 1, 2, 2, 1],
        width_factor=[1]*4 + [2.75]
    )


def create_RepVGG_B1():
    return RepVGG(
        filter_depth=[1, 4, 6, 16, 1],
        filter_list=[16, 16, 32, 64, 64],
        stride=[1, 1, 2, 2, 1],
        width_factor=[2]*4 + [4]
    )

def create_RepVGG_B2():
    return RepVGG(
        filter_depth=[1, 4, 6, 16, 1],
        filter_list=[16, 16, 32, 64, 64],
        stride=[1, 1, 2, 2, 1],
        width_factor=[2.5]*4 + [5]
    )


def create_RepVGG_B3():
    return RepVGG(
        filter_depth=[1, 4, 6, 16, 1],
        filter_list=[16, 16, 32, 64, 64],
        stride=[1, 1, 2, 2, 1],
        width_factor=[3]*4 + [5]
    )


if __name__ == "__main__":

    # model = RepVGG(
    #     filter_depth=[1, 2, 4, 14, 1],
    #     filter_list=[16, 16, 32, 64, 128],
    #     stride=[1, 1, 2, 2, 1],
    #     width_factor=[0.75,0.75,0.75,0.75,2.5]
    # )

    model = create_RepVGG_A1()
    # QA
    model.eval()

    input = torch.randn(1, 3, 32, 32)

    out_train = model(input)

    deployed_model = deploy_model(model)

    # print(deployed_model)

    out_eval = deployed_model(input)

    print(((out_train - out_eval) ** 2).sum())
