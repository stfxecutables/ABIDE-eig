# fmt: off
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
logging.getLogger("tensorflow").setLevel(logging.FATAL)
logging.getLogger("tensorboard").setLevel(logging.FATAL)
# fmt: on


from pathlib import Path

import torch
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    Softplus,
)


class Lin(Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.model = Sequential(
            Linear(in_channels, out_channels, bias=True),
            ReLU(),
            BatchNorm1d(out_channels),
            Dropout(0.0),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x


class SoftLinear(Module):
    def __init__(self, in_channels: int, out_channels: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.model = Sequential(
            Linear(in_channels, out_channels, bias=True),
            Softplus(),
            BatchNorm1d(out_channels),
            Dropout(0.0),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x


class PointLinear(Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.model = Sequential(
            Linear(4, out_channels, bias=True),
            LeakyReLU(),
            BatchNorm1d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_mean = torch.mean(x, dim=1)
        x_min = torch.min(x, dim=1)[0]
        x_max = torch.max(x, dim=1)[0]
        x_sd = torch.std(x, dim=1)
        x = torch.stack([x_mean, x_min, x_max, x_sd], dim=1)
        x = self.model(x)
        return x


class Conv(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.model = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                # padding="same",
                padding=0,
                bias=True,
            ),
            LeakyReLU(),
            BatchNorm1d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # needs x.shape == (B, C, seq_length)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.model(x)
        return x


class Conv2(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.model = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                # padding="same",
                padding=0,
                bias=True,
            ),
            LeakyReLU(),
            BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # needs x.shape == (B, C, seq_length)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.model(x)
        return x


class GlobalAveragePool1D(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, C, len)
        return torch.mean(x, dim=-1)


class GlobalAveragePool2D(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, C, H, W)
        return torch.mean(x, dim=(2, 3))
