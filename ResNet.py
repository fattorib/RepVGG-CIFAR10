import torch
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.nn.init as init


def _weights_init(m):
    classname = m.__class__.__name__
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


# ----------Building Blocks----------


class ResidualBlock(nn.Module):
    # One full block of a given filter size
    def __init__(self, in_filters, out_filters, N, downsample=True):
        super(ResidualBlock, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.N = N
        self.downsample = downsample
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                self.in_filters,
                self.in_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_filters),
            nn.ReLU(),
            nn.Conv2d(
                self.in_filters,
                self.in_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_filters),
        )

        self.residual_block = nn.ModuleList(
            [copy.deepcopy(self.conv_block) for _ in range(self.N - 1)]
        )
        self.bn = nn.BatchNorm2d(out_filters, affine=True)
        # Downsample using stride of (2,2)
        if self.downsample:
            self.final_block = nn.Sequential(
                nn.Conv2d(
                    self.in_filters,
                    self.in_filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.in_filters),
                nn.ReLU(),
                nn.Conv2d(
                    self.in_filters,
                    self.out_filters,
                    kernel_size=3,
                    stride=(2, 2),
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.out_filters),
            )
        else:
            self.final_block = nn.Sequential(
                nn.Conv2d(
                    self.in_filters,
                    self.in_filters,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.in_filters),
                nn.ReLU(),
                nn.Conv2d(
                    self.in_filters,
                    self.in_filters,
                    kernel_size=3,
                    stride=(1, 1),
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(self.in_filters),
            )
        self.apply(_weights_init)

    def forward(self, x):
        for block in self.residual_block:
            residual = x
            x = block(x)
            x += residual
            x = F.relu(x)
        # Perform downsampling on last layer and add final residual
        residual = x
        x = self.final_block(x)
        if self.downsample:
            x += self.pad_identity(residual)
        else:
            # Don't downsample on final layer
            x += residual
        x = F.relu(x)
        return x

    def pad_identity(self, x):
        # Perform padding on filters to allow final residual connections
        return F.pad(
            x[:, :, ::2, ::2],
            (0, 0, 0, 0, self.out_filters // 4, self.out_filters // 4),
            "constant",
            0,
        )


class SimpleBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(SimpleBlock, self).__init__()
        self.in_filters = in_filters
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                self.in_filters,
                self.in_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_filters),
            nn.ReLU(),
            nn.Conv2d(
                self.in_filters,
                self.in_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_filters),
        )
        # self.apply(_weights_init)

    def forward(self, x):
        out = self.conv_block(x)
        out += x
        return F.relu(out)

    def pad_identity(self, x):
        # Perform padding on filters to allow final residual connections
        return F.pad(
            x[:, :, ::2, ::2],
            (0, 0, 0, 0, self.out_filters // 4, self.out_filters // 4),
            "constant",
            0,
        )


class DownsampleBlock(nn.Module):
    def __init__(self, in_filters, out_filters):
        super(DownsampleBlock, self).__init__()
        self.in_filters = in_filters
        self.out_filters = out_filters
        self.conv_block = nn.Sequential(
            nn.Conv2d(
                self.in_filters,
                self.in_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.in_filters),
            nn.ReLU(),
        )
        self.downsample = nn.Sequential(
            nn.Conv2d(
                self.in_filters,
                self.out_filters,
                kernel_size=3,
                stride=(2, 2),
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(self.out_filters),
        )

        # self.apply(_weights_init)

    def forward(self, x):
        out = self.conv_block(x)
        out = self.downsample(x)
        out += self.pad_identity(x)
        return F.relu(out)

    def pad_identity(self, x):
        # Perform padding on filters to allow final residual connections
        return F.pad(
            x[:, :, ::2, ::2],
            (0, 0, 0, 0, self.out_filters // 4, self.out_filters // 4),
            "constant",
            0,
        )


# ----------ResNets----------


class ResNet(nn.Module):
    def __init__(self, filters_list, N):
        super(ResNet, self).__init__()
        self.filters_list = filters_list
        self.N = N
        self.first_layer = nn.Conv2d(
            3, filters_list[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(filters_list[0])
        self.first_block = ResidualBlock(
            in_filters=self.filters_list[0], out_filters=self.filters_list[1], N=self.N
        )

        self.second_block = ResidualBlock(
            in_filters=self.filters_list[1], out_filters=self.filters_list[2], N=self.N
        )

        self.third_block = ResidualBlock(
            in_filters=self.filters_list[2],
            out_filters=self.filters_list[2],
            N=self.N,
            downsample=False,
        )
        self.fc = nn.Linear(64, 10)

        self.apply(_weights_init)

    def forward(self, x):
        x = F.relu(self.bn(self.first_layer(x)))
        x = self.first_block(x)
        x = self.second_block(x)
        x = self.third_block(x)

        # Global average pooling
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class ResNetV2(nn.Module):
    def __init__(self, filters_list, N):
        super(ResNetV2, self).__init__()
        self.filters_list = filters_list
        self.N = N
        self.first_layer = nn.Conv2d(
            3, filters_list[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn = nn.BatchNorm2d(filters_list[0])
        self.first_block_type = SimpleBlock(self.filters_list[0], self.filters_list[0])
        self.second_block_type = SimpleBlock(self.filters_list[1], self.filters_list[2])
        self.third_block_type = SimpleBlock(self.filters_list[2], self.filters_list[2])

        # First block
        self.first_block = nn.ModuleList(
            [copy.deepcopy(self.first_block_type) for _ in range(self.N - 1)]
        )
        self.first_downsample = DownsampleBlock(
            self.filters_list[0], self.filters_list[1]
        )

        self.second_block = nn.ModuleList(
            [copy.deepcopy(self.second_block_type) for _ in range(self.N - 1)]
        )
        self.second_downsample = DownsampleBlock(
            self.filters_list[1], self.filters_list[2]
        )

        self.third_block = nn.ModuleList(
            [copy.deepcopy(self.third_block_type) for _ in range(self.N)]
        )

        self.apply(_weights_init)

        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.bn(self.first_layer(x)))
        for block in self.first_block:
            x = block(x)
        x = self.first_downsample(x)

        for block in self.second_block:
            x = block(x)

        x = self.second_downsample(x)

        for block in self.third_block:
            x = block(x)

        # Global average pooling
        x = F.avg_pool2d(x, x.size()[3])
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


if __name__ == "__main__":

    model = ResNet(filters_list=[16, 32, 64], N=3)
