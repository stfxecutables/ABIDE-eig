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
from tqdm import tqdm
from typing_extensions import Literal

from src.analysis.predict.deep_learning.constants import INPUT_SHAPE
from src.analysis.predict.deep_learning.models.layers import InputConv


def test_input_shapes(capsys: CaptureFixture) -> None:
    K = [1, 2, 3, 4, 5, 7]
    S = [1, 2]
    P = [1, 2, 3, 4, 5]
    D = [1, 2, 3]
    G = [1, 2, 3, 4]
    x = torch.rand([1, *INPUT_SHAPE], device="cuda")

    with capsys.disabled():
        for k in tqdm(K, leave=True, desc="K"):
            for s in tqdm(S, leave=False, desc="S"):
                for p in tqdm(P, leave=False, desc="P"):
                    for d in tqdm(D, leave=False, desc="D"):
                        for depthwise in [True, False]:
                            if depthwise:
                                for g in G:
                                    conv = InputConv(k, s, p, d, depthwise, g)
                                    conv.conv.to(device="cuda")
                                    out = conv(x)
                                    exp = (1, *conv.output_shape())
                                    assert out.shape == exp
                            else:
                                conv = InputConv(k, s, p, d, depthwise)
                                conv.conv.to(device="cuda")
                                out = conv(x)
                                exp = (1, *conv.output_shape())
                                assert out.shape == exp
