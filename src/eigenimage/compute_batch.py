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
from pathlib import Path
from typing import List
from typing_extensions import Literal

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants import NII_PATH
from src.eigenimage.compute import compute_eigenimage


SHAPES = NII_PATH / "shapes.json"

# we'll be cutting the timepoints to the last 176, which means computations
# take very close to 2 hours each time, so 11 computations per 24h window,
# i.e. 11 computations per batch
T_LENGTH = 176
BATCH_SIZE = 11


def get_batch_idx() -> int:
    parser = ArgumentParser()
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    return int(args.batch)


def get_files() -> List[Path]:
    df = pd.read_json(SHAPES)
    tr_2 = df.loc[df.dt == 2.0, :].sort_index()
    files = [NII_PATH / file for file in tr_2.index.to_list()]
    return files


if __name__ == "__main__":
    files = get_files()
    idx = get_batch_idx()
    batch = files[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
    for file in batch:
        compute_eigenimage(file, covariance=True, t=T_LENGTH)
