from argparse import Namespace
from typing import Any

import torch
from torch.nn.modules.padding import ConstantPad3d

from src.analysis.predict.deep_learning.constants import INPUT_SHAPE, PADDED_SHAPE
from src.analysis.predict.deep_learning.models.layers.conv import MultiConv4D
from src.analysis.predict.deep_learning.models.layers.utils import EVEN_PAD


def test_shape() -> None:
    x = torch.rand([1, 1, 8, 6, 6, 6], device="cuda")
    conv = MultiConv4D(
        in_channels=1,
        channel_expansion=2,
        spatial_in_shape=x.shape[3:],
        temporal_in_shape=x.shape[2],
        spatial_kernel=3,
        spatial_dilation=1,
        temporal_kernel=5,
        temporal_dilation=1,
    ).to("cuda")
    res = conv(x)
    assert res.shape == x.shape


def test_print(capsys: Any) -> None:
    with capsys.disabled():
        conv = MultiConv4D(
            in_channels=1,
            channel_expansion=2,
            spatial_in_shape=(16, 32, 32),
            temporal_in_shape=120,
            spatial_kernel=3,
            spatial_dilation=1,
            temporal_kernel=5,
            temporal_dilation=1,
        )
        print()
        print(conv)


def test_memory(capsys: Any) -> None:
    defaults = dict(
        spatial_kernel=3,
        spatial_stride=1,
        spatial_dilation=1,
        spatial_in_shape=PADDED_SHAPE[1:],
        temporal_kernel=5,
        temporal_stride=2,
        temporal_dilation=1,
        spatial_padding="same",
        temporal_padding=0,
    )
    with capsys.disabled():
        x = torch.rand([1, 1, *INPUT_SHAPE], device="cuda")
        print("\nCreated x")
        model = torch.nn.Sequential(
            ConstantPad3d(EVEN_PAD, 0),
            MultiConv4D(in_channels=1, channel_expansion=4, temporal_in_shape=175, **defaults),
            MultiConv4D(
                in_channels=4, channel_expansion=2, temporal_in_shape=175 // 2 - 1, **defaults
            ),
            MultiConv4D(in_channels=8, channel_expansion=2, temporal_in_shape=41, **defaults),
            MultiConv4D(in_channels=16, channel_expansion=1, temporal_in_shape=19, **defaults),
        ).to("cuda")
        print("Created model")
        res = model(x)
        print(res)
