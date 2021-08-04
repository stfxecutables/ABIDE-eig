from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from torch.nn import Conv3d, LSTMCell, Module, ReLU
from typing_extensions import Literal


class GlobalAveragePooling(Module):
    def __init__(self, dims: Union[int, Tuple[int, ...]] = (2, 3, 4), keepdim: bool = False):
        super().__init__()
        self.dims = dims
        self.keepdim = bool(keepdim)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        return torch.mean(x, dim=self.dims, keepdim=self.keepdim)

    def __str__(self) -> str:
        return f"{self.__class__.__name__} reducing_dims={self.dims})"

    __repr__ = __str__
