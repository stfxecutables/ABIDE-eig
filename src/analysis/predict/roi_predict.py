# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
from src.run.cc_setup import setup_environment  # isort:skip
setup_environment()
# fmt: on

import logging
import os
from pprint import pprint
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from tqdm import tqdm
from typing_extensions import Literal

from src.analysis.predict.hypertune import HtuneResult, hypertune_classifier
from src.analysis.predict.roi_predict import identity, max, roi_dataframes
from src.constants.paths import DATA, ROIS

if not ROIS.exists():
    os.makedirs(ROIS, exist_ok=True)
    os.makedirs(ROIS / "ctrl", exist_ok=True)
    os.makedirs(ROIS / "autism", exist_ok=True)
EIGIMGS = DATA / "eigimgs"
ATLAS_DIR = DATA / "atlases"
ATLAS = ATLAS_DIR / "cc400_roi_atlas_ALIGNED.nii.gz"
LEGEND = ATLAS_DIR / "CC400_ROI_labels.csv"
LOG = os.environ.get("CC_CLUSTER") is None
VERBOSITY = optuna.logging.INFO if LOG else optuna.logging.ERROR


def predict_from_full_rois(
    source: Literal["func", "eigimg"] = "eigimg",
    norm: Literal["div", "diff"] = "div",
    slicer: slice = slice(None),
    slice_reducer: Callable[[ndarray], ndarray] = identity,
    weight_sharing: Literal["brains", "rois", "voxels"] = "rois",
) -> DataFrame:
    pass


def predict_from_roi_reductions(
    source: Literal["func", "eigimg"] = "eigimg",
    norm: Literal["div", "diff"] = "div",
    reducer: Callable[[ndarray], ndarray] = None,
    reducer_name: str = None,
    slicer: slice = slice(None),
    slice_reducer: Callable[[ndarray], ndarray] = identity,
    weight_sharing: Literal["brains", "per-roi", "rois", "voxels"] = "per-roi",
    classifier: Type = SVC,
    classifier_args: Dict[str, Any] = {},
) -> Tuple[DataFrame, float, Optional[HtuneResult]]:
    """Predict autism vs. control from various roi reductions with various classifiers and/or models

    Parameters
    ----------
    weight_sharing: Literal["brains", "rois", "voxels"] = "rois"
        If "brains", fits one gigantic model across all subjects, rois, and voxels, i.e.
        X.shape == (n_ROIs * (N_autism + N_ctrl), T). Class is predicted based on a single time
        series.

        If "rois", fits one model per ROI, i.e. 400 models, so X.shape is for each ROI model
        X.shape == (N_autism + N_ctrl, T). The 400 predictions are combined by majority vote.

        If "voxels", fits one model per voxel, and combines as in the "rois" case.

    Returns
    -------
    val1: Any
    """
    if weight_sharing not in ["per-roi", "rois"]:
        raise NotImplementedError()

    autism, ctrl, names = roi_dataframes(
        source=source,
        norm=norm,
        reducer=reducer,
        reducer_name=reducer_name,
        slicer=slicer,
        slice_reducer=slice_reducer,
    )
    autism, ctrl = autism.T, ctrl.T
    autism["target"] = 1
    ctrl["target"] = 0
    df = pd.concat([autism, ctrl], axis=0, ignore_index=True)
    guess = np.max([len(autism), len(ctrl)]) / len(df)
    if weight_sharing == "per-roi":
        scores = DataFrame(index=names, columns=["acc"])
        for roi, name in tqdm(enumerate((names)), total=len(names), desc="Fitting SVM for ROI"):
            X = np.stack(df.iloc[:, roi].to_numpy())
            X = MinMaxScaler().fit_transform(X)
            y = df["target"].to_numpy()
            res = cross_val_score(classifier(**classifier_args), X, y, cv=5, scoring="accuracy")
            scores.loc[name, "acc"] = np.mean(res)
        return scores, guess, None
    elif weight_sharing == "rois":
        Xs = []
        for roi, name in enumerate(names):
            Xs.append(np.stack(df.iloc[:, roi].to_numpy()))
        X = np.concatenate(Xs, axis=1)
        # X = StandardScaler().fit_transform(X)
        y = df["target"].to_numpy()
        if LOG:
            print(f"Cross-validating {classifier} on X with shape {X.shape} with args:")
            pprint(classifier_args, indent=2)
        htune_result = hypertune_classifier(
            "rf",
            X,
            y,
            n_trials=200,
            cv_method=5,
            verbosity=VERBOSITY,
        )
        # res = cross_val_score(classifier(**classifier_args), X, y, cv=5, scoring="accuracy")
        logging.debug(f"Best val_acc: {htune_result.val_acc}")
        scores = pd.DataFrame(index=["all"], columns=["acc"], data=htune_result.val_acc)
        return scores, guess, htune_result
    else:
        raise ValueError("Invalid `weight_sharing` option")


if __name__ == "__main__":
    scores, guess, htuned = predict_from_roi_reductions(
        source="func",
        norm="div",
        reducer=max,
        # slicer=slice(126, None),
        slice_reducer=identity,
        weight_sharing="rois",
        classifier=RandomForestClassifier,
        classifier_args=dict(n_jobs=-1),
    )
    print(f"Mean acc: {np.round(np.mean(scores), 3).item()}  (guess = {np.round(guess, 3)})")
    print(
        f"CI: ({np.round(np.percentile(scores, 5), 3)}, {np.round(np.percentile(scores, 95), 3)})"
    )

"""
Guess = 0.569
Results: Func
    Best Acc: 0.582
        source="func", norm=None, reducer=std,
        slicer=slice(None), slice_reducer=identity,
        weight_sharing="rois", classifier=RandomForestClassifier,
    Best Acc: 0.584
        source="func", norm=None, reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        weight_sharing="rois", classifier=RandomForestClassifier,
    Best Acc: 0.613
        source="func", norm="div", reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        weight_sharing="rois", classifier=RandomForestClassifier,

TODO: test PCA reduced ROIs and/or PCA reduced fMRI images
Results: Eigimg
    Best Acc: 0.637
        source="eigimg", norm="div", reducer=std,
        slicer=slice(None), slice_reducer=identity,
        weight_sharing="rois", classifier=RandomForestClassifier,
    Best Acc: 0.637  # also faster to run, despite less data
        source="eigimg", norm="div", reducer=std,
        slicer=slice(90, 160), slice_reducer=identity,
        weight_sharing="rois", classifier=RandomForestClassifier,
    Best Acc: 0.629  # !!!
        source="eigimg", norm="div", reducer=std,
        slicer=slice(125, 129), slice_reducer=identity,
        weight_sharing="rois", classifier=RandomForestClassifier,
    Best Acc: 0.64
        source="eigimg", norm="div", reducer=std,
        slicer=slice(126, None), slice_reducer=identity,
        weight_sharing="rois", classifier=RandomForestClassifier,
"""
