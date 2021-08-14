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
from torch import Tensor
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


def model_fits_gpu(x: Tensor, batch: int, T: int, n_layer: int, hidden: int) -> Optional[DataFrame]:
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
        print(
            f"Success with: batch_size={batch}, T={T}, hidden_size={hidden}, num_layers={n_layer}"
        )
        return DataFrame(dict(batch=batch, T=T, hidden_size=hidden, num_layers=n_layer), index=[0])
    except:
        return None


def test_mem(capsys: CaptureFixture) -> None:
    OUTFILE = Path(__file__).resolve().parent / "memtest.json"
    hidden = HIDDEN_START = 4
    batch = BATCH_START = 2
    n_layer = N_LAYER_START = 1
    T_START = 20
    T = INPUT_SHAPE[0]
    T = T_START
    dfs = []
    # We have a bunch of params: batch, T, n_layer, hidden. We start each of
    # these at the lowest *acceptable* values, and then search through any
    # combinations of where *at least one value* of the four is larger than
    # the starting value at that index. *If* we find a value v = (b, t, n, h) that
    # leads to a fail, then for any 4-tuple `tup` if np.all(np.array(tup) >= np.array(v)),
    # then the args in `tup` *also* cause a fail.
    prev = curr = np.array([batch, T, n_layer, hidden])
    while T < 100:
        n_layer = N_LAYER_START
        # prev = curr = np.array([batch, T, n_layer, hidden])
        while n_layer < 5:
            batch = BATCH_START
            while batch < 8:
                hidden = HIDDEN_START
                while hidden < 33:
                    if np.all(curr > prev):
                        continue
                    x = torch.rand((batch, T, 1, *SPATIAL)).to(device="cuda")
                    with capsys.disabled():
                        df = model_fits_gpu(x, batch, T, n_layer, hidden)
                    if df is not None:
                        dfs.append(df)
                        pd.concat(dfs, axis=0, ignore_index=True).to_json(OUTFILE, indent=2)
                        hidden += 1
                        curr = prev = np.array([batch, T, n_layer, hidden])
                    else:
                        prev = np.array([batch, T, n_layer, hidden])
                        break
                batch += 1
            n_layer += 1
        T += 5
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_json(Path(__file__).resolve().parent / "memtest.json")
    with capsys.disabled():
        print(df)


# Success with: batch_size=1, hidden_size=1, num_layers=1
# Success with: batch_size=1, hidden_size=2, num_layers=1
# Success with: batch_size=1, hidden_size=3, num_layers=1
# Success with: batch_size=1, hidden_size=4, num_layers=1
# Success with: batch_size=2, hidden_size=1, num_layers=1
# Success with: batch_size=2, hidden_size=2, num_layers=1
# Success with: batch_size=3, hidden_size=1, num_layers=1
# Success with: batch_size=4, hidden_size=1, num_layers=1
