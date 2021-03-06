import torch
from torch.nn.modules import padding
import torchvision
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import copy
import torch.nn.init as init
from torch import Tensor


def reparam_func(layer):
    """[summary]

    Args:
        layer: Single RepVGG block

        Returns the reparamitrized weights
    """

    # 3x3 weight fuse
    std = (layer.bn_3.running_var + layer.bn_3.eps).sqrt()
    t = (layer.bn_3.weight / std).reshape(-1, 1, 1, 1)

    reparam_weight_3 = layer.conv_3.weight * t
    reparam_bias_3 = layer.bn_3.bias - layer.bn_3.running_mean * layer.bn_3.weight / std

    reparam_weight = reparam_weight_3
    reparam_bias = reparam_bias_3

    # 1x1 weight fuse
    std = (layer.bn_1.running_var + layer.bn_1.eps).sqrt()
    t = (layer.bn_1.weight / std).reshape(-1, 1, 1, 1)

    reparam_weight_1 = layer.conv_1.weight * t
    reparam_bias_1 = layer.bn_1.bias - layer.bn_1.running_mean * layer.bn_1.weight / std

    reparam_weight += F.pad(reparam_weight_1, [1, 1, 1, 1], mode="constant", value=0)
    reparam_bias += reparam_bias_1

    if layer.conv_3.weight.shape[0] == layer.conv_3.weight.shape[1]:
        # Check if in/out filters are equal, if not, we skip the identity reparam
        if hasattr(layer, "bn_0"):

            # idx weight fuse - we only have access to bn_0
            std = (layer.bn_0.running_var + layer.bn_0.eps).sqrt()
            t = (layer.bn_0.weight / std).reshape(-1, 1, 1, 1)

            channel_shape = layer.conv_3.weight.shape

            idx_weight = (
                torch.eye(channel_shape[0], channel_shape[0])
                .unsqueeze(2)
                .unsqueeze(3)
                .to(layer.conv_3.weight.device)
            )

            reparam_weight_0 = idx_weight * t

            reparam_bias_0 = (
                layer.bn_0.bias - layer.bn_0.running_mean * layer.bn_0.weight / std
            )

            reparam_weight += F.pad(
                reparam_weight_0, [1, 1, 1, 1], mode="constant", value=0
            )
            reparam_bias += reparam_bias_0

    assert reparam_weight.shape == layer.conv_3.weight.shape

    return reparam_weight, reparam_bias
