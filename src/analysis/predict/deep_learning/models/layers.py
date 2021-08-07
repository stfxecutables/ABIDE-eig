from math import floor
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from torch.nn import (
    AdaptiveMaxPool3d,
    BatchNorm3d,
    BCEWithLogitsLoss,
    ConstantPad3d,
    Conv3d,
    Linear,
    LSTMCell,
    Module,
    PReLU,
    ReLU,
)
from typing_extensions import Literal

from src.analysis.predict.deep_learning.constants import INPUT_SHAPE

Tuple3d = Union[int, Tuple[int, int, int]]

# need to pad (47, 59, 42) to (48, 60, 42)
EVEN_PAD = (0, 0, 1, 0, 1, 0)


def outsize_3d(s: int, kernel: int, dilation: int, stride: int, padding: int) -> int:
    """Get the output size for a dimension of size `s`"""
    p, d = padding, dilation
    k, r = kernel, stride
    return floor((s + 2 * p - d * (k - 1) - 1) / r + 1)


# see https://discuss.pytorch.org/t/same-padding-equivalent-in-pytorch/85121/4
# for padding logic
def padding_same_3d(
    input_shape: Tuple[int, int, int], kernel: int, dilation: int
) -> Tuple[int, int, int, int, int, int]:
    """Assumes symmetric / cubic kernels, dilations, and stride=1"""

    def pad(width: int, k: int) -> Tuple[int, int]:
        p = max(k - 1, 0) if (width % 1) == 0 else max(k - (width % 1), 0)
        p_top = p // 2
        p_bot = p - p_top
        return p_top, p_bot

    k = kernel + (kernel - 1) * (dilation - 1)  # effective kernel size
    D, H, W = input_shape
    return (*pad(D, k), *pad(H, k), *pad(W, k))


class Conv3dSame(Module):
    STRIDE = 1

    def __init__(
        self,
        in_channels: int,
        spatial_in: Tuple[int, int, int],
        out_channels: int,
        kernel_size: int,
        dilation: int,
        depthwise: bool = False,
        depthwise_groups: int = 1,
    ):
        super().__init__()
        self.kernel = kernel_size
        self.dilation = dilation
        self.groups = in_channels if depthwise else 1
        self.d_groups = depthwise_groups if depthwise else 1
        self.in_channels = in_channels
        self.out_channels = in_channels * self.d_groups
        self.pad = ConstantPad3d(padding_same_3d(spatial_in, kernel_size, dilation), 0)
        self.conv = Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=self.STRIDE,
            dilation=dilation,
            groups=self.d_groups,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.pad(x)
        x = self.conv(x)
        return x


class GlobalAveragePooling(Module):
    def __init__(self, dims: Union[int, Tuple[int, ...]] = (2, 3, 4), keepdim: bool = False):
        super().__init__()
        self.dims = dims
        self.keepdim = bool(keepdim)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(x, dim=self.dims, keepdim=self.keepdim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} reducing_dims={self.dims})"

    __repr__ = __str__


class Downsample(Module):
    def __init__(self) -> None:
        super().__init__()


class InputConv(Module):
    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        depthwise: bool,
        depthwise_groups: int = 1,
    ) -> None:
        super().__init__()
        ch = in_channels
        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = ch if depthwise else 1
        self.d_groups = depthwise_groups if depthwise else 1
        self.in_channels = ch
        self.out_channels = ch * self.d_groups
        self.padder = ConstantPad3d(EVEN_PAD, 0)
        self.conv = Conv3d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=self.groups,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.padder(x)
        x = self.conv(x)
        return x

    def output_shape(self) -> Tuple[int, int, int, int]:
        unpadded = INPUT_SHAPE[1:]  # S = "spatial"
        S = (unpadded[0] + 1, unpadded[1] + 1, unpadded[2])
        spatial_out = tuple(map(self.outsize, S))
        return (self.out_channels, *spatial_out)

    def outsize(self, s: int) -> int:
        """Get the output size for a dimension of size `s`"""
        p, d = self.padding, self.dilation
        k, r = self.kernel, self.stride
        return outsize_3d(s, k, d, r, p)


class ResBlock(Module):
    def __init__(self) -> None:
        super().__init__()
        ch = INPUT_SHAPE[0]
        KERNEL = 3
        STRIDE = 2
        DILATION = 3
        PADDING = 3
        conv_args: Dict = dict(
            kernel_size=KERNEL,
            stride=STRIDE,
            dilation=DILATION,
            padding=PADDING,
            bias=False,
            groups=1,
        )
        # BNorm  after PReLU
        # see https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
        self.conv1 = Conv3d(in_channels=ch, out_channels=ch, **conv_args)
        self.relu1 = PReLU()
        self.norm1 = BatchNorm3d(ch)
        self.conv2 = Conv3d(in_channels=ch, out_channels=ch, **conv_args)
        self.relu2 = PReLU()
        self.norm2 = BatchNorm3d(ch)
