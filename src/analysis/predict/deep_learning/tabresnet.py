"""
We make tiny modifications to
https://github.com/pytorch/vision/blob/7d955df73fe0e9b47f7d6c77c699324b256fc41f/torchvision/models/resnet.py
to use Linear layers
"""

from typing import Any, Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import GELU, AdaptiveAvgPool1d, BatchNorm1d, Conv1d, Linear, Sequential


class WideTabResLayer(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.lin1 = Linear(in_features, out_features, bias=False)
        self.bn1 = BatchNorm1d(out_features)
        self.lin2 = Linear(out_features, out_features, bias=False)
        self.bn2 = BatchNorm1d(out_features)
        self.gelu = GELU()
        self.expand = Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor
        identity = x
        out = self.lin1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.lin2(out)
        out = self.bn2(out)
        out = self.gelu(out)

        identity = identity.unsqueeze(2)
        identity = self.expand(identity).squeeze()
        out += identity
        out = self.gelu(out)
        return out


class TabInput(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin = Linear(in_features, out_features, bias=False)
        self.bn = BatchNorm1d(out_features)
        self.act = GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = x.squeeze()
        x = self.bn(x)
        x = self.act(x)
        # x = x.unsqueeze(1)
        return x


class TabWideResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        width: int = 64,
        n_layers: int = 4,
    ) -> None:
        super().__init__()
        exp = 2
        self.width = width
        self.input = TabInput(in_features, width)
        layers = []
        for d in range(n_layers):
            r_in, r_out = 2 ** d, 2 ** (d + 1)
            layers.append(WideTabResLayer(self.width * r_in, self.width * r_in))
            layers.append(WideTabResLayer(self.width * r_in, self.width * r_out))
        self.res_layers = Sequential(*layers)
        self.out = Linear(self.width * r_out, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        x = self.res_layers(x)
        # x = torch.flatten(x)
        x = self.out(x)
        return x


if __name__ == "__main__":
    x = torch.rand([2, 175], device="cuda")
    model = TabWideResNet(in_features=175).to(device="cuda")
    out = model(x)
    print(out.shape)
