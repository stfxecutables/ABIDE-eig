"""
We make tiny modifications to
https://github.com/pytorch/vision/blob/7d955df73fe0e9b47f7d6c77c699324b256fc41f/torchvision/models/resnet.py
to use 3d convolutions and GroupNorm instead, for the small batches.

Note: GroupNorm degrades to LayerNorm in the 1-channel case: see  see Fig 2. of the original
GroupNorm paper https://arxiv.org/pdf/1803.08494.pdf) or PyTorch documentation for GroupNorm

I.e. for int `n`:

    GroupNorm(num_groups=g, num_channels=n*g) is normal group norm for any `n`
    GroupNorm(num_groups=n, num_channels=n) is InstanceNorm
    GroupNorm(num_groups=1, num_channels=n) is LayerNorm
"""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


def conv3x3x3(
    in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1
) -> nn.Conv3d:
    """3x3x3 convolution with padding"""
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv3d:
    """1x1 convolution"""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.GroupNorm
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        layers: List[int],
        num_classes: int = 1000,
        groups: int = 1,
        width_per_group: int = 64,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.init_ch = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.init_ch, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = norm_layer(self.init_ch)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(self.init_ch, blocks=layers[0], stride=1)
        self.layer2 = self._make_layer(self.init_ch * 2, blocks=layers[1], stride=2)
        self.layer3 = self._make_layer(self.init_ch * 4, blocks=layers[2], stride=2)
        self.layer4 = self._make_layer(self.init_ch * 8, blocks=layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.init_ch * 8, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(
        self,
        planes: int,
        blocks: int,
        stride: int = 1,
    ) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        if stride != 1 or self.init_ch != planes:
            downsample = nn.Sequential(
                conv1x1x1(self.init_ch, planes, stride),
                norm_layer(planes),
            )
        shared_args: Dict = dict(
            groups=self.groups,
            base_width=self.base_width,
            dilation=self.dilation,
            norm_layer=norm_layer,
        )
        layers = [
            BasicBlock(
                inplanes=self.init_ch,
                planes=planes,
                stride=stride,
                downsample=downsample,
                **shared_args,
            )
        ]
        self.init_ch = planes
        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    inplanes=self.init_ch,
                    planes=planes,
                    stride=1,
                    downsample=None,
                    **shared_args,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)


def _resnet(layers: List[int], **kwargs: Any) -> ResNet:
    return ResNet(layers, **kwargs)


def resnet18(**kwargs: Any) -> ResNet:
    """ResNet18-3d"""
    return _resnet([2, 2, 2, 2], **kwargs)


def resnet34(**kwargs: Any) -> ResNet:
    """ResNet34-3d"""
    return _resnet([3, 4, 6, 3], **kwargs)
