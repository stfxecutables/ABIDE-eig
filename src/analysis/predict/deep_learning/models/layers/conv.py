from math import floor
from typing import Dict, Tuple, Union, cast

from torch import Tensor
from torch.nn import (
    BatchNorm3d,
    BCEWithLogitsLoss,
    ConstantPad3d,
    Conv3d,
    Dropout,
    GroupNorm,
    InstanceNorm3d,
    LayerNorm,
    Linear,
    MaxPool3d,
    Module,
    PReLU,
    ReLU,
    Sequential,
)
from torch.nn.modules.pooling import MaxPool3d
from typing_extensions import Literal

from src.analysis.predict.deep_learning.constants import INPUT_SHAPE
from src.analysis.predict.deep_learning.models.layers.cbam import CBAM
from src.analysis.predict.deep_learning.models.layers.utils import (
    EVEN_PAD,
    norm_layer,
    outsize_3d,
    padding_same,
)


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
        self.pad = ConstantPad3d(padding_same(spatial_in, kernel_size, dilation), 0)
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


class InputConv(Module):
    """Take in fMRI images and pad to even.

    Parameters
    ----------
    depthwise: bool
        If True, uses depthwise convolutions, i.e. will set `groups` argument
        to `in_channels`, and `out_channels` to `k * in_channels` where `k` is
        specified in `depthwise_groups` arg.

    depthwise_k: int = 1
        See above notes for `depthwise` arg.
    """

    def __init__(
        self,
        in_channels: int,
        kernel_size: int,
        stride: int,
        padding: int,
        dilation: int,
        depthwise: bool,
        depthwise_k: int = 1,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.groups = self.in_channels if depthwise else 1
        self.out_channels = self.in_channels * depthwise_k if depthwise else self.in_channels

        self.kernel = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

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
        return (self.out_channels, *spatial_out)  # type: ignore

    def outsize(self, s: int) -> int:
        """Get the output size for a dimension of size `s`"""
        p, d = self.padding, self.dilation
        k, r = self.kernel, self.stride
        return outsize_3d(s, k, d, r, p)  # type: ignore


class ResBlock3d(Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        residual: bool = True,
        halve: bool = True,
        depthwise: bool = False,
        depthwise_factor: int = None,
        norm: Literal["batch", "group", "instance", "layer"] = "batch",
        norm_groups: int = 5,
        cbam: bool = False,
        cbam_reduction: int = 4,
    ) -> None:
        """Summary

        Parameters
        ----------
        depthwise_factor: int = None
            Applies only if `depthwise=True`. Must be an integer greater than 1 or None. Results in:
                out_channels = depthwise_factor * in_channels
                groups = depthwise_factor
            If None, results in:
                out_channels = in_channels
                groups = in_channels
        """
        super().__init__()
        self.in_channels = ci = in_channels
        if depthwise and (depthwise_factor is None):
            self.out_channels = in_channels
            self.groups = self.out_channels
        elif depthwise and (depthwise_factor is not None):
            self.out_channels = depthwise_factor * in_channels
            self.groups = depthwise_factor
        else:
            self.out_channels = out_channels
            self.groups = 1
        co = self.out_channels

        self.kernel = kernel_size
        self.dilation = dilation
        self.residual = residual
        self.halve = halve
        # padding for same is effective kernel size // 2
        self.padding = (self.kernel + (self.kernel - 1) * (self.dilation - 1)) // 2
        self.norm_groups = norm_groups
        self.norm = norm
        self.is_cbam = cbam
        self.cbam_reduction = cbam_reduction

        conv_args: Dict = dict(
            out_channels=self.out_channels,
            kernel_size=self.kernel,
            stride=1,
            dilation=self.dilation,
            padding=self.padding,
            groups=self.groups,
            bias=False,
        )
        # BNorm  after PReLU
        # see https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
        self.conv1 = Conv3d(in_channels=ci, **conv_args)
        self.relu1 = PReLU()
        self.norm1 = norm_layer(self.norm, in_channels=self.out_channels, groups=norm_groups)
        self.conv2 = Conv3d(in_channels=co, **conv_args)
        self.relu2 = PReLU()
        self.norm2 = norm_layer(self.norm, in_channels=self.out_channels, groups=norm_groups)
        if halve:
            self.pool = MaxPool3d(2, 2)
        if self.is_cbam:
            self.cbam = CBAM(in_channels=self.out_channels, reduction=self.cbam_reduction)

    def forward(self, x: Tensor) -> Tensor:
        # we use modern identity mapping here in the hopes of better performance
        # https://arxiv.org/abs/1603.05027
        out: Tensor
        identity = x
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.norm1(out)
        out = self.conv2(out)
        if self.is_cbam:
            out = self.cbam(out)
        out = self.relu2(out)
        out = self.norm2(out)
        out += identity  # we are assuming padding="same"
        if self.halve:
            out = self.pool(out)
        return out

    def output_shape(self, input_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
        def outsize(insize: int) -> int:
            # pooling layer has no padding, dilation so is straightforward
            k = s = 2  # kernel size and stride
            numerator = insize - (k - 1) - 1
            return floor(numerator / s + 1)

        if not self.halve:
            return input_shape
        # now calculate size from pooling
        return tuple(map(outsize, input_shape))  # type: ignore


# This implements CBAM after a ResNet ResBlock, e.g. something like Fig 3 of
# https://arxiv.org/abs/1807.06521, or see also the code for that paper, e.g.
# https://github.com/Jongchan/attention-module/blob/master/MODELS/model_resnet.py#L17-L51
class CBAMResBlock(Module):
    """Inserts sequential channel and spatial attention layers into a ResNet-style residual block."""

    def __init__(self, in_channels: int, out_channels: int, cbam: bool = True, reduction: int = 4):
        super().__init__()
        self.has_cbam = cbam
        self.res = Sequential(
            Conv3d(in_channels, out_channels, 3, padding=1, bias=False),
            ReLU(inplace=True),
            BatchNorm3d(out_channels),
            Conv3d(out_channels, out_channels, 3, padding=1, bias=False),
        )
        self.relu = ReLU(inplace=True)
        self.bnorm = BatchNorm3d(out_channels)
        if cbam:
            self.cbam = CBAM(in_channels=in_channels, reduction=reduction)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        x = self.res(x)
        if self.has_cbam:
            x = self.cbam(x)
        x += identity
        x = self.relu(x)
        return x
