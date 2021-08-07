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
from src.analysis.predict.deep_learning.models.layers import Conv3dSame, InputConv, ResBlock3d

SPATIAL = INPUT_SHAPE[1:]
TEST_SHAPE = (1, 1, *SPATIAL)  # no need for so many channels
PADDED_TEST_SHAPE = (1, 1, 48, 60, 42)


def test_input_shapes(capsys: CaptureFixture) -> None:
    K = [1, 2, 3, 4, 5, 7]
    S = [1, 2]
    P = [1, 2, 3, 4, 5]
    D = [1, 2, 3]
    G = [1, 2, 3, 4]
    x = torch.rand(TEST_SHAPE, device="cuda")

    with capsys.disabled():
        for k in tqdm(K, leave=True, desc="K"):
            for s in tqdm(S, leave=False, desc="S"):
                for p in tqdm(P, leave=False, desc="P"):
                    for d in tqdm(D, leave=False, desc="D"):
                        for depthwise in [True, False]:
                            if depthwise:
                                for g in G:
                                    conv = InputConv(1, k, s, p, d, depthwise, g)
                                    conv.conv.to(device="cuda")
                                    out = conv(x)
                                    exp = (1, *conv.output_shape())
                                    assert out.shape == exp
                            else:
                                conv = InputConv(1, k, s, p, d, depthwise)
                                conv.conv.to(device="cuda")
                                out = conv(x)
                                exp = (1, *conv.output_shape())
                                assert out.shape == exp


def test_padding(capsys: CaptureFixture) -> None:
    K = [1, 2, 3, 4, 5, 7]
    D = [1, 2, 3]
    EVENS = [1, 1, 32, 32, 32]
    ODDS = [1, 1, 33, 33, 33]
    MIXED = [1, 1, 32, 33, 33]

    for shape in [EVENS, ODDS, MIXED]:
        x = torch.rand(shape, device="cuda")
        with capsys.disabled():
            for k in tqdm(K, leave=True, desc="K"):
                for d in tqdm(D, leave=False, desc="D"):
                    conv = Conv3dSame(
                        in_channels=1,
                        spatial_in=shape[2:],
                        out_channels=1,
                        kernel_size=k,
                        dilation=d,
                    )
                    conv.to(device="cuda")
                    out = conv(x)
                    assert out.shape == x.shape

def test_resblock(capsys: CaptureFixture) -> None:
    res = ResBlock3d(1, 1, kernel_size=3, dilation=3)  # effective kernel size of 7
    x = torch.rand((1, 1, 16, 16, 16))
    out = res(x)
    assert out.shape == (1, 1, 8, 8, 8)
