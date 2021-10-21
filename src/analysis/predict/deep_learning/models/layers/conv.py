from math import floor, prod
from typing import Dict, List, Tuple, Union, cast

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm3d,
    BCEWithLogitsLoss,
    ConstantPad3d,
    Conv1d,
    Conv3d,
    Dropout,
    Flatten,
    GroupNorm,
    InstanceNorm3d,
    LayerNorm,
    Linear,
    MaxPool3d,
    Module,
    ModuleList,
    PReLU,
    ReLU,
    Sequential,
)
from torch.nn.modules.pooling import MaxPool3d
from tqdm import tqdm
from typing_extensions import Literal

from src.analysis.predict.deep_learning.constants import INPUT_SHAPE
from src.analysis.predict.deep_learning.models.layers.cbam import CBAM
from src.analysis.predict.deep_learning.models.layers.utils import (
    EVEN_PAD,
    norm_layer,
    outsize_3d,
    padding_same,
)

Padding = Union[Literal["same"], int]


class TemporalFlatten(Module):
    """Convert all spatial dimensions to channels, i.e. a shape (B, C_new, T) so that Conv1d
    is happy"""

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        """x.shape = (B, C, T, *SPATIAL)"""
        B, C, T, H, W, D = x.shape
        spatial = (H, W, D)
        x = x.permute([0, 2, 3, 4, 5, 1])
        x = x.reshape([B, C * prod(spatial), T])
        return x


class TemporalUnflatten(Module):
    def __init__(self, out_channels: int, spatial_shape: Tuple[int, ...]) -> None:
        super().__init__()
        self.spatial_shape = spatial_shape
        self.out_channels = out_channels

    def forward(self, x: Tensor) -> Tensor:
        B, N, T = x.shape
        x = x.reshape([B, self.out_channels, *self.spatial_shape, T])
        x = x.permute([0, 1, 5, 2, 3, 4])  # put T back in place
        return x


# see https://github.com/pytorch/vision/blob/9e474c3c46c0871838c021093c67a9c7eb1863ea/torchvision/models/video/resnet.py#L36
class MultiConv4D(Sequential):
    """Expects inputs of shape (B, C, T, *SPATIAL)

    Notes 2
    -------
    Conv(3+1)D won't really work, in that it won't allow channel expansions. So we do something
    similar in spirit, where multiple depthwise separable Conv3D layers are used to mimic the multiple channels,
    i.e. the depthwise separability leaves timepoints un-perturbed, and multiple such conv layers can be concatenated
    back together to duplicate the actual channels. Then one gigantic pointwise conv1d handles the channel expansion.

    Notes
    -----
    To make this efficient, our spatial convolution (Conv3D) needs to treat timepoints as channels, and use
    depthwise-separable convolutions so that the same convolution is applied to all timepoints. Then, because
    there is no nice Conv4D so we can just do a hacky Conv4d(kernel_size=(k, 1, 1, 1)) trick, we have to do
    a Conv1D on a flattened image (!!). So we do some trickery with channels and re-interpeting dimensions,
    but it should work...

    Since out inputs have shape (B, C, T, *SPATIAL), after the first spatial conv they have shape
    (B, mid_channels, T, *SPATIAL_R), where *SPATIAL_R depends on padding, stride, dilation, etc.

    """

    def __init__(
        self,
        in_channels: int,
        channel_expansion: int,
        spatial_in_shape: Tuple[int, int, int],
        temporal_in_shape: int,
        spatial_kernel: int = 3,
        spatial_stride: int = 1,
        spatial_dilation: int = 1,
        temporal_kernel: int = 5,
        temporal_stride: int = 1,
        temporal_dilation: int = 1,
        spatial_padding: Padding = "same",
        temporal_padding: Padding = "same",
    ):
        super().__init__()
        self.in_channels: int = in_channels
        self.channel_expansion: int = channel_expansion
        self.spatial_in_shape: Tuple[int, int, int] = spatial_in_shape
        self.spatial_kernel: int = spatial_kernel
        self.spatial_stride: int = spatial_stride
        self.spatial_dilation: int = spatial_dilation
        self.temporal_in_shape: int = temporal_in_shape
        self.temporal_kernel: int = temporal_kernel
        self.temporal_stride: int = temporal_stride
        self.temporal_dilation: int = temporal_dilation
        self.spatial_padding: Padding = spatial_padding
        self.temporal_padding: Padding = temporal_padding
        self.s_padding, self.t_padding = self.init_paddings()

        self.separables = ModuleList(
            [
                Conv3d(
                    self.temporal_in_shape,
                    self.temporal_in_shape,
                    kernel_size=self.spatial_kernel,
                    stride=self.spatial_stride,
                    padding=self.s_padding,
                    bias=False,
                    groups=self.temporal_in_shape,
                )
                for _ in range(self.in_channels)
            ]
        )
        # self.pointwise = Conv3d(
        #     in_channels=self.in_channels,
        #     out_channels=self.mid_channels,
        #     kernel_size=1,
        #     stride=1,
        #     bias=False,
        # )
        self.norms = ModuleList(
            [BatchNorm3d(self.temporal_in_shape) for _ in range(self.in_channels)]
        )
        self.relu = ReLU(inplace=False)
        self.flatten = TemporalFlatten()
        spatial_out = self.spatial_outshape()
        in_ch = self.in_channels * prod(spatial_out)
        out_ch = self.channel_expansion * in_ch
        self.temporal = Conv1d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=self.temporal_kernel,
            stride=self.temporal_stride,
            padding=self.t_padding,
            groups=in_ch,
            bias=False,
        )
        self.unflatten = TemporalUnflatten(self.channel_expansion * self.in_channels, spatial_out)
        self.final_norms = ModuleList(
            [
                BatchNorm3d(self.temporal_outshape())
                for _ in range(self.in_channels * self.channel_expansion)
            ]
        )

    def forward(self, x: Tensor) -> Tensor:
        """x.shape == (B, C, T, *SPATIAL)"""
        out: List[Tensor] = []
        for i in range(x.shape[1]):
            cout = self.separables[i](x[:, i])  # conv out
            cout = self.relu(cout)
            cout = self.norms[i](cout)
            cout = cout.unsqueeze(1)  # add back channel dim
            out.append(cout)  # add back channel dim
        x = torch.cat(out, dim=1)  # type: ignore # along channel dim
        del out
        # for i in range(len(out)):
        #     del out[i]
        del cout
        # x = self.separable(x)
        # x = self.pointwise(x)
        # x = self.norm(x)
        # x = self.relu(x)
        x = self.flatten(x)
        x = self.temporal(x)
        x = self.unflatten(x)
        x = self.relu(x)
        for i in range(x.shape[1]):
            x[:, i] = self.final_norms[i](x[:, i])
        return x

    def spatial_outshape(self) -> Tuple[int, int, int]:
        if self.spatial_padding == "same":
            return self.spatial_in_shape
        return tuple(  # type: ignore
            [
                outsize_3d(
                    s,
                    self.spatial_kernel,
                    self.spatial_dilation,
                    self.spatial_stride,
                    self.spatial_padding,
                )
                for s in self.spatial_in_shape
            ]
        )

    def temporal_outshape(self) -> int:
        if self.temporal_padding == "same":
            return self.temporal_in_shape
        return outsize_3d(  # type: ignore
            self.temporal_in_shape,
            self.temporal_kernel,
            self.temporal_dilation,
            self.temporal_stride,
            self.temporal_padding,
        )

    def outshape(self) -> Tuple[int, int, int, int]:
        t = self.temporal_outshape()
        spatial = self.spatial_outshape()
        return (t, *spatial)

    def init_paddings(self) -> Tuple[int, int]:
        s_padding: int = -1
        t_padding: int = -1
        if self.spatial_padding == "same":
            if self.spatial_stride > 1:
                raise NotImplementedError(
                    "Currently padding only available for symmetric stride of 1"
                )
            if self.spatial_kernel // 2 == 0:
                raise ValueError("Even kernels require asymmetric padding")
            s_padding = padding_same(
                self.spatial_in_shape, self.spatial_kernel, self.spatial_dilation
            )[
                ::2
            ]  # odd kernel means we only need 3 padding values
        else:
            self.s_padding = int(self.spatial_padding)
        if self.temporal_padding == "same":
            if self.temporal_stride > 1:
                raise NotImplementedError(
                    "Currently padding only available for symmetric stride of 1"
                )
            if self.temporal_kernel // 2 == 0:
                raise ValueError("Even kernels require asymmetric padding")
            t_padding = padding_same(
                (self.temporal_in_shape,), self.temporal_kernel, self.temporal_dilation
            )[
                ::2
            ]  # odd kernel means we only need 3 padding values
        else:
            t_padding = int(self.temporal_padding)
        return s_padding, t_padding


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
