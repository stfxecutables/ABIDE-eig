from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

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
