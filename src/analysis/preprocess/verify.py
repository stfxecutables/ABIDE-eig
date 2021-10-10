import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm
from typing_extensions import Literal

# fmt: off
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
from src.analysis.preprocess.constants import FEATURES_DIR

# fmt: on


def _show_data_shapes(parent: Path) -> Tuple[ndarray, ndarray]:
    files = sorted(parent.rglob("*"))
    exclude = [FEATURES_DIR.name, "cc200", "cc400"]
    parents = np.unique([f.parent for f in files])
    dfs = []
    for p in tqdm(parents):
        if p.name in exclude:
            continue
        label = f"{p.parent.name}/{p.name}"
        if "features_cpac" in label:
            label = label[label.find("/") + 1 :]
        files = p.rglob("*.npy")
        arrs = [np.load(f) for f in files]
        shapes, counts = np.unique([str(a.shape) for a in arrs], return_counts=True)
        # print(f"\n{label}")
        # print(f"{'shape':^12}|{'count':^7}")
        # print(f"{'-'*(12+7+ 1 + 1)}")
        for shape, count in zip(shapes, counts):
            dfs.append(DataFrame(dict(feature=label, shape=shape, count=count), index=[0]))
            # print(f"{shape:^12}|{count:>7}")
    df = pd.concat(dfs, axis=0, ignore_index=True)
    print(df.to_markdown(tablefmt="simple", index=False))
    pd.options.display.max_rows = 200
    print(df.groupby(["feature", "shape"]).count())


if __name__ == "__main__":
    _show_data_shapes(FEATURES_DIR)
