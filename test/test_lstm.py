from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
    cast,
    final,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
import torch
from _pytest.capture import CaptureFixture
from numpy import ndarray
from pandas import DataFrame, Series
from torch.nn import Conv3d
from tqdm import tqdm
from typing_extensions import Literal

from src.analysis.predict.deep_learning.constants import INPUT_SHAPE
from src.analysis.predict.deep_learning.models.layers.lstm import ConvLSTM3d

SPATIAL = INPUT_SHAPE[1:]
TEST_SHAPE = (1, 1, *SPATIAL)  # no need for so many channels
PAD_SPATIAL = (48, 60, 42)
PADDED_TEST_SHAPE = (1, 1, *PAD_SPATIAL)
CONVLSTM_SHAPE = (1, 75, 1, *PAD_SPATIAL)  # (B, T, C, *SPATIAL)


def test_sanity(capsys: CaptureFixture) -> None:
    HIDDEN = 8
    x = torch.rand(CONVLSTM_SHAPE, device="cuda")
    with capsys.disabled():
        lstm = ConvLSTM3d(
            in_channels=1,
            in_spatial_dims=PAD_SPATIAL,
            num_layers=1,
            hidden_sizes=HIDDEN,
            kernel_sizes=3,
            dilations=2,
        )
        lstm.to(device="cuda")
        hidden, cell = lstm(x)
        assert hidden.shape == (1, HIDDEN, *PAD_SPATIAL)
        assert cell.shape == (1, HIDDEN, *PAD_SPATIAL)


def test_stacked(capsys: CaptureFixture) -> None:
    HIDDEN = 4
    x = torch.rand(CONVLSTM_SHAPE, device="cuda")
    with capsys.disabled():
        lstm = ConvLSTM3d(
            in_channels=1,
            in_spatial_dims=PAD_SPATIAL,
            num_layers=2,
            hidden_sizes=HIDDEN,
            kernel_sizes=3,
            dilations=2,
        )
        lstm.to(device="cuda")
        hidden, cell = lstm(x)
        assert hidden.shape == (1, HIDDEN, *PAD_SPATIAL)
        assert cell.shape == (1, HIDDEN, *PAD_SPATIAL)


def test_mem(capsys: CaptureFixture) -> None:
    n_layer = batch = hidden = 1
    T = INPUT_SHAPE[0]
    while n_layer < 3:
        batch = 1
        while batch < 12:
            x = torch.rand((batch, T, 1, *SPATIAL)).to(device="cuda")
            hidden = 1
            while hidden < 40:
                try:
                    lstm = ConvLSTM3d(
                        in_channels=1,
                        in_spatial_dims=SPATIAL,
                        num_layers=n_layer,
                        hidden_sizes=hidden,
                        kernel_sizes=3,
                        dilations=2,
                        # depthwise=True,
                    )
                    lstm.to(device="cuda")
                    lstm(x)
                    with capsys.disabled():
                        print(
                            f"Success with: batch_size={batch}, hidden_size={hidden}, num_layers={n_layer}"
                        )
                except:
                    pass
                hidden += 1
            batch += 1
        n_layer += 1


# Success with: batch_size=1, hidden_size=1, num_layers=1
# Success with: batch_size=1, hidden_size=2, num_layers=1
# Success with: batch_size=1, hidden_size=3, num_layers=1
# Success with: batch_size=1, hidden_size=4, num_layers=1
# Success with: batch_size=2, hidden_size=1, num_layers=1
# Success with: batch_size=2, hidden_size=2, num_layers=1
# Success with: batch_size=3, hidden_size=1, num_layers=1
# Success with: batch_size=4, hidden_size=1, num_layers=1
