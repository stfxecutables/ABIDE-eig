from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
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
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

N = 1_000
T_MIN, T_MAX = 80, 500

if __name__ == "__main__":
    Ts = list(range(T_MIN, T_MAX))
    corrs = []
    ts = []
    for t in tqdm(Ts, total=len(Ts), desc="Correlating"):
        for i in range(N):
            x = np.random.standard_normal([2, t])
            corrs.append(np.corrcoef(x, rowvar=True)[0, 1])
            ts.append(t)
    sbn.set_style("darkgrid")
    fig, ax = plt.subplots()
    ax.scatter(ts, corrs, color="black")
    ax.set_xlabel("T")
    ax.set_ylabel("r")
    ax.set_title("Correlations (r) per Sequence Length (T)")
    plt.show()
