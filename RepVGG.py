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


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")



class RepVGGBlock(nn.Module):
    """Single RepVGG block. We build these into distinct 'stages'"""

    def __init__(self, in_channels, out_channels):
        super(RepVGGBlock, self).__init__()

        self.conv_3 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            bias=False,
        )
        self.bn_3 = nn.BatchNorm2d(num_features=out_channels)

        self.conv_1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(1, 1),
            padding=0,
            stride=1,
            bias = False
        )
        self.bn_1 = nn.BatchNorm2d(num_features=out_channels)

        self.identity = False
        if in_channels == out_channels:
            # Use identity branch
            self.bn_0 = nn.BatchNorm2d(num_features=out_channels)
            self.identity = True

        self.apply(_weights_init)

        # Reparam conv block
        self.rep_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=1,
            bias=True,
        )
        self.reparam = False

    def forward(self, x):
        if not self.reparam:
            x_3 = self.bn_3(self.conv_3(x))
            x_1 = self.bn_1(self.conv_1(x))

            if self.identity:
                x_0 = self.bn_0(x)

                return F.relu(x_3 + x_1 + x_0)

            else:
                return F.relu(x_3 + x_1)
        else:
            
            return F.relu(self.rep_conv(x))


class DownsampleRepVGGBlock(nn.Module):
    """Downsample RepVGG block. Comes at the end of a stage"""

    def __init__(self, num_channels):
        super(DownsampleRepVGGBlock, self).__init__()

        self.conv_3 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=(2, 2),
            bias=False,
        )
        self.bn_3 = nn.BatchNorm2d(num_features=num_channels)

        self.conv_1 = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(1, 1),
            padding=0,
            stride=(2, 2),
            bias = False
        )
        self.bn_1 = nn.BatchNorm2d(num_features=num_channels)

        # We pad the identity the usual way
        self.bn_0 = nn.BatchNorm2d(num_features=num_channels)

        self.apply(_weights_init)

        # Reparam conv block
        self.rep_conv = nn.Conv2d(
            in_channels=num_channels,
            out_channels=num_channels,
            kernel_size=(3, 3),
            padding=1,
            stride=(2, 2),
            bias=True,
        )
        self.reparam = False

    def forward(self, x):
        if not self.reparam:
            x_3 = self.bn_3(self.conv_3(x))
            x_1 = self.bn_1(self.conv_1(x))

            x_0 = self.bn_0(x[:, :, ::2, ::2])

            return F.relu(x_3 + x_1 + x_0)
        else:
            
            return F.relu(self.rep_conv(x))


class RepVGGStage(nn.Module):
    """Single RepVGG stage. These are stacked together to form the full RepVGG architecture"""

    def __init__(self, in_channels, out_channels, N, a, b=None):
        super(RepVGGStage, self).__init__()

        self.in_channels = in_channels if in_channels == 3 else int(a * in_channels)
        self.out_channels = (
            int(a * out_channels) if b is None else int(b * out_channels)
        )

        self.sequential = nn.Sequential(
            *[RepVGGBlock(in_channels=self.in_channels, out_channels=self.out_channels)]
            + [
                RepVGGBlock(
                    in_channels=self.out_channels, out_channels=self.out_channels
                )
                for _ in range(0, N - 2)
            ]
            + [DownsampleRepVGGBlock(num_channels=self.out_channels)]
        )

    def forward(self, x):
        return self.sequential(x)



    def _reparam(self):
        with torch.no_grad():
            for stage in self.sequential:
                # print(stage)
                reparam_weight, reparam_bias = reparam_func(stage)
                stage.rep_conv.weight.data = reparam_weight
                stage.rep_conv.bias.data = reparam_bias
                stage.reparam = True

    def _train(self):
        with torch.no_grad():
            for stage in self.sequential:
                stage.reparam = False

                


class RepVGG(nn.Module):
    def __init__(
        self,
        filter_depth=[1, 2, 4, 14, 1],
        filter_list=[64, 64, 128, 256, 512],
        a=1,
        b=1,
    ):
        super(RepVGG, self).__init__()

        self.stages = nn.Sequential(
            *[
                RepVGGStage(
                    in_channels=3,
                    out_channels=filter_list[0],
                    N=filter_depth[0],
                    a=a,
                )
            ]
            + [
                RepVGGStage(
                    in_channels=filter_list[i - 1],
                    out_channels=filter_list[i],
                    N=filter_depth[i],
                    a=a,
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

    def _train(self):
        for stage in self.stages:
            stage._train()

            


if __name__ == "__main__":

    #Something still isn't right here. Unstable when network is very deep
    model = RepVGG(
        filter_depth=[1, 4, 8],
        filter_list=[16, 32, 64],
        a = 0.75,
        b = 2.5
    )

    # QA
    model.eval()

    input = torch.randn(1,3,32,32)

    out_train = model(input)

    model._reparam()

    out_eval = model(input)

    model._use_train_branches()

    print(((out_train - out_eval) ** 2).sum())


    out_train_2 = model(input)

    print(((out_train - out_train_2) ** 2).sum())

    # model = RepVGGStage(in_channels=64,out_channels=512,N = 5, a= 1)

    # model.eval()


    # input = torch.randn(1, 64, 32, 32)

    # out_train = model(input)

    # model._reparam()


    # out_eval = model(input)

    # print(((out_train - out_eval) ** 2).sum())


    # model = RepVGGBlock(in_channels=64, out_channels=128)

    # # model = DownsampleRepVGGBlock(num_channels=64)

    # model.eval()


    # input = torch.randn(1, 64, 32, 32)

    # out_train = model(input)

    # reparam_weight, reparam_bias = reparam_func(model)
    # model.rep_conv.weight.data = reparam_weight
    # model.rep_conv.bias.data = reparam_bias

    # model.reparam = True

    # out_eval = model(input)

    # print(((out_train - out_eval) ** 2).sum())
