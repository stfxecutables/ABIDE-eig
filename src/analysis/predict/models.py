# fmt: off
import sys  # isort:skip
from pathlib import Path # isort: skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from math import floor

import optuna
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

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

if __name__ == "__main__":
    features = [f for f in FEATURES if len(f.shape_data.shape) == 1]
    fnames, accs, norms, smins, smaxs = [], [], [], [], []
    for f in tqdm(features, leave=True):
        for norm in tqdm([NormMethod.S_MINMAX, NormMethod.F_MINMAX, NormMethod.F_STD], leave=True):
            for slice_min in tqdm([0, 0.25, 0.50, 0.75, 0.9], leave=False):
                for slice_max in [0.50, 0.75, 0.9, 1.0]:
                    if slice_max <= slice_min:
                        continue
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
                    accs.append(result["acc"])
                    norms.append(norm.value)
                    smins.append(slice_min)
                    smaxs.append(slice_max)
                    fnames.append(f.name)
    df = DataFrame(
        data={
            "feature": fnames,
            "norm": norms,
            "slice_start": smins,
            "slice_end": smaxs,
            "acc": accs,
        }
    )
    df.to_parquet(ROOT / "htune_initial_results_LDA.parquet")
    pd.options.display.max_rows = 999
    print(df.sort_values(by="acc", ascending=False))
