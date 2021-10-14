# fmt: off
import sys  # isort:skip
from pathlib import Path # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from math import floor
from typing import Dict, Optional

import optuna
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import ParameterGrid
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.analysis.features import FEATURES, NormMethod
from src.analysis.predict.hypertune import (
    evaluate_hypertuned,
    evaluate_untuned,
    hypertune_classifier,
)

"""
Notes
-----
We have a few feature types:

* multi-channel 1D
    - ROI means, ROI sds, their concatenation
* 2D matrix
    - r_mean, r_sd, and their concatenation
* 1D non-sequential (single or multi-channel)
    - r_mean, r_sd, r_desc (multichannel)
    - Lap, Lap_concat, eig_r_mean, eig_r_sd, and concat
* 1D sequential
    - Laplacian and plain eigenvalues, and their concatenations

Classic ML: SVM, LR, RF AdaBoostDTree
Deep Learn: MLP, Conv1D, ...?
"""


def fit_to_args(args: Dict) -> Optional[Dict]:
    f = args["feature"]
    norm = args["norm"]
    slice_min = args["slice_min"]
    slice_max = args["slice_max"]
    if slice_max <= slice_min:
        return None
    # print(f"Fitting {f} using normalization method {norm}")
    x, y = f.load(norm)
    smin = floor(slice_min * x.shape[1])
    smax = floor(slice_max * x.shape[1]) if slice_max != 1.0 else None
    sl = slice(smin, smax)
    x = x[:, sl]
    # htune_result = hypertune_classifier(
    #     "lda", x, y, n_trials=1, verbosity=optuna.logging.ERROR
    # )
    # result = evaluate_hypertuned(htune_result, 5, x, y, log=False)
    result = evaluate_untuned("rf", 5, x, y, log=False)
    return {
        "feature": f.name,
        "norm": norm.value,
        "smin": slice_min,
        "smax": slice_max,
        "acc": result["acc"],
    }


if __name__ == "__main__":
    features = [f for f in FEATURES if len(f.shape_data.shape) == 1]
    # current grid is about 20-30 minutes for random forest, no tuning, 8 workers
    # in tqdm, -1 workers in RF
    grid = list(
        ParameterGrid(
            dict(
                feature=features,
                norm=[NormMethod.S_MINMAX, NormMethod.F_MINMAX, NormMethod.F_STD],
                slice_min= [0, 0.25, 0.50, 0.75, 0.9],
                slice_max=[0.50, 0.75, 0.9, 1.0],
                # slice_min=[0],
                # slice_max=[0.50],
            )
        )
    )
    results = process_map(fit_to_args, grid, max_workers=8)
    results = [r for r in results if r is not None]
    dfs = [DataFrame(info, index=[0]) for info in results]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_parquet(ROOT / "htune_initial_results_LDA.parquet")
    pd.options.display.max_rows = 999
    print(df.sort_values(by="acc", ascending=False))
