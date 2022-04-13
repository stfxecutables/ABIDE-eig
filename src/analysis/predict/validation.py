from __future__ import annotations  # isort:skip # noqa

import os
import sys
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from itertools import repeat
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import float64 as f64
from numpy import ndarray
from numpy.random import MT19937, RandomState, SeedSequence
from numpy.typing import ArrayLike, NDArray
from pandas import DataFrame, Series
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

FloatArray = NDArray[np.floating]

# fmt: off
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

# HACK for local testing
if os.environ.get("CC_CLUSTER") is None:
    os.environ["CC_CLUSTER"] = "home"


from src.analysis.features import FEATURES, SUBJ_DATA, Feature, NormMethod
from src.analysis.predict.hypertune import evaluate_untuned
from src.analysis.preprocess.atlas import Atlas

f64Array = NDArray[f64]
Arrays = List[f64Array]

"""Define splitting procedures for reproducible and sane validation of ABIDE data

Notes
-----
Must implement:

    - stratified splitting by site
    - leave-one-site-out
    - repeatable (robust) k-fold
"""


def show_kfold_variability(n_reps: int = 100, k: int = 5, stratify: bool = False) -> DataFrame:
    site = SUBJ_DATA.site.astype(str)
    sites = site.unique()
    site_ids = np.arange(len(sites))
    site_to_idx = {s: i for i, s in enumerate(sites)}
    idx_to_site = {i: s for i, s in enumerate(sites)}

    strat = SUBJ_DATA.DX_GROUP.astype(str) + SUBJ_DATA.site.astype(str)
    X = pd.DataFrame(
        dict(
            site=site.apply(lambda s: site_to_idx[s]).to_numpy(), aut=SUBJ_DATA.DX_GROUP.to_numpy()
        )
    )
    dfs = []
    zeros: List[str] = []
    ones: List[str] = []
    threes: List[str] = []
    for i in tqdm(range(n_reps)):
        # trains, tests = list(zip(*StratifiedKFold(shuffle=True).split(X, y=strat)))
        folder = StratifiedKFold if stratify else KFold
        trains, tests = list(zip(*folder(n_splits=K, shuffle=True).split(X, y=strat)))
        tests = [X.loc[test] for test in tests]
        for i, test in enumerate(tests):
            for sid in site_ids:
                idx = test.loc[:, "site"] == sid
                count = np.sum(idx)
                aut_ratio = np.mean(test.loc[idx, "aut"])
                dfs.append(
                    pd.DataFrame(
                        {"Site": sid, "N": count, "fold": i + 1, "ratio": aut_ratio}, index=[0]
                    )
                )
                if count == 0:
                    zeros.append(idx_to_site[sid])
                elif count == 1:
                    ones.append(idx_to_site[sid])
                elif count <= 3:
                    threes.append(idx_to_site[sid])
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.Site = df.Site.apply(lambda s: idx_to_site[s])
    pd.options.display.max_rows = 100
    site_counts = df.drop(columns="fold").groupby(["Site"])
    print(site_counts.describe())
    print(f"Number of folds one or more site is excluded from testing: {len(zeros)}")
    print(
        f"Number of folds where one or more sites have only one subject in one of the test folds: {len(ones)}"
    )
    print(
        f"Number of folds where one or more sites have <= 3 subjects in one of the test folds: {len(threes)}"
    )
    print(
        f"Times each site was excluded from a fold ({n_reps}@{k} folds = {k * n_reps} folds total):"
    )
    sites, count = np.unique(zeros, return_counts=True)
    table = pd.DataFrame({"Site": sites, "Excluded": count})
    print(table)
    print("Base counts")
    print(SUBJ_DATA.groupby(["site"]).count().rename(columns={"DX_GROUP": "Count"}).T)


def cv_scores(args: Namespace) -> float:
    feature = args.feature
    rng = RandomState(MT19937(args.rng))
    X, y, g = feature.load(normalize=NormMethod.F_STD)
    cv = StratifiedKFold(K, shuffle=True, random_state=rng) if args.stratify else KFold(K, shuffle=True, random_state=rng)
    return np.mean(
        cross_val_score(
            SVC(),
            X,
            y,
            scoring="accuracy",
            cv=cv,
        )
    )


if __name__ == "__main__":
    SUBJ_DATA
    N_REPS = 100
    K = 5
    STRATIFY = True
    # show_kfold_variability(n_reps=N_REPS, k=K, stratify=STRATIFY)

    feature = list(filter(lambda f: f.name == "lap_mean04", FEATURES))[0]
    classifier = "svm"
    rngs = np.random.SeedSequence().spawn(N_REPS)

    args = [Namespace(**dict(feature=feature, classifier=classifier, rng=rng, stratify=STRATIFY)) for rng in rngs]
    scores = process_map(cv_scores, args)
    print(pd.DataFrame(scores).describe(percentiles=[0.05, 0.10, 0.25, 0.5, 0.75, 0.9, 0.95]).T)
