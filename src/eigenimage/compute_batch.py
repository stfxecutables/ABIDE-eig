import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants import NII_PATH
from src.eigenimage.compute import compute_eigenimage


NIIS = sorted(NII_PATH.glob("*minimal.nii.gz"))
INFO = pd.read_json(NII_PATH).drop(columns=["H", "W", "D"])
SHAPES = NII_PATH / "shapes.json"
T_LENGTH = 176


def get_batch_idx() -> int:
    parser = ArgumentParser()
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    return int(args.batch)


if __name__ == "__main__":
    df = pd.read_json(SHAPES)
    tr_2 = df.loc[df.dt == 2.0, :].sort_index()
    files = [NII_PATH / file for file in tr_2.index.to_list()]
    # we'll be cutting the timepoints to the last 176, which means computations
    # take very close to 2 hours each time, so 11 computations per 24h window,
    # i.e. 11 computations per batch
    idx = get_batch_idx()
    batch = files[idx * 11 : (idx + 1) * 11]
    for file in batch:
        compute_eigenimage(file, covariance=True, t=T_LENGTH)
