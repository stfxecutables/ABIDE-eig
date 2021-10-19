# fmt: off
import sys  # isort:skip
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from math import floor
from typing import Any, Dict, List, Optional

import numpy as np
import optuna
import pandas as pd
import torch
from pandas import DataFrame
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import LearningRateMonitor
from sklearn.model_selection import GroupShuffleSplit, ParameterGrid, StratifiedShuffleSplit
from torch.nn import Module
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.analysis.features import FEATURES, Feature, NormMethod
from src.analysis.predict.deep_learning.tabresnet import TabLightningNet
from src.analysis.predict.hypertune import (
    Classifier,
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
    classifier = args["classifier"]
    f: Feature = args["feature"]
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
    htune_result = hypertune_classifier(
        classifier, x, y, n_trials=20, verbosity=optuna.logging.INFO
    )
    result = evaluate_hypertuned(htune_result, 5, x, y, log=False)
    # result = evaluate_untuned(classifier, 5, x, y, log=False)
    return {
        "feature": f.name,
        "atlas": f.atlas.name if f.atlas is not None else "",
        "norm": norm.value,
        "smin": slice_min,
        "smax": slice_max,
        "acc": result["acc"],
    }


def features_sanity_test(features: List[Feature], classifier: Classifier) -> None:
    # current grid is about 20-30 minutes for random forest, no tuning, 8 workers
    grid = list(
        ParameterGrid(
            dict(
                classifier=[classifier],
                feature=features,
                norm=[NormMethod.S_MINMAX, NormMethod.F_MINMAX, NormMethod.F_STD],
                # slice_min= [0, 0.25, 0.50, 0.75, 0.9],
                # slice_max=[0.50, 0.75, 0.9, 1.0],
                slice_min=[0],
                slice_max=[1.00],
            )
        )
    )
    results = process_map(fit_to_args, grid, max_workers=1)
    results = [r for r in results if r is not None]
    dfs = [DataFrame(info, index=[0]) for info in results]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_parquet(ROOT / f"htune_initial_results_{classifier}.parquet")
    pd.options.display.max_rows = 999
    print(df.sort_values(by="acc", ascending=False))


def resnet_sanity_test(features: List[Feature], slice_min=0.0, slice_max=1.0) -> None:
    for f in features:
        # norm = NormMethod.F_STD
        norm = NormMethod.S_MINMAX
        # norm = NormMethod.F_MINMAX
        x, y, groups = f.load(norm)
        smin = floor(slice_min * x.shape[1])
        smax = floor(slice_max * x.shape[1]) if slice_max != 1.0 else None
        sl = slice(smin, smax)
        x = x[:, sl]
        x, y = torch.from_numpy(x).float(), torch.from_numpy(y).float()
        stratifier = [f"{site}{target}" for site, target in zip(y, groups)]
        # stratifier = groups
        tr, val = next(
            StratifiedShuffleSplit(n_splits=1, test_size=200).split(
                np.arange(x.shape[0]), y=stratifier
            )
        )
        x_train, y_train = x[tr], y[tr]
        x_val, y_val = x[val], y[val]
        val_dummy = torch.mean(y_val).item()
        if val_dummy < 0.5:
            val_dummy = 1 - val_dummy
        train_data = TensorDataset(x_train, y_train)
        val_data = TensorDataset(x_val, y_val)
        train_loader = DataLoader(
            train_data, batch_size=32, shuffle=True, num_workers=8, drop_last=True
        )
        val_loader = DataLoader(val_data, batch_size=25, shuffle=False, num_workers=8)
        model = TabLightningNet(
            in_features=x.shape[1], width=32, n_layers=2, dropout=0.2, val_dummy=val_dummy
        )
        atlas = f.atlas.name if f.atlas is not None else ""
        print("=" * 120)
        print(
            f"Beginning training for {f.name} ({atlas}) sliced to [{smin}:{smax}) normalization {norm.value}"
        )
        print(f)
        print("=" * 120)
        outdir = ROOT / "resnet_sanity_test"
        outdir /= f.name if atlas == "" else f"{f.name}/{atlas}"
        outdir /= norm.name
        trainer = Trainer(
            gpus=1,
            max_epochs=2000,
            default_root_dir=outdir,
            callbacks=[LearningRateMonitor()],
            progress_bar_refresh_rate=34,  # batches per epoch
        )
        trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    CLASSIFIER = "xgb"

    features = [f for f in FEATURES if len(f.shape_data.shape) == 1 and "eig_mean" in f.name][0]
    # features_sanity_test(features, CLASSIFIER)
    # even tiny slices (0.95 to 1.0) result in identical performance to full data...
    resnet_sanity_test([features], slice_min=0.00, slice_max=1.0)
