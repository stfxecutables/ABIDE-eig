import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import nibabel as nib
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

if os.environ.get("CC_CLUSTER") is not None:
    SCRATCH = os.environ["SCRATCH"]
    os.environ["MPLCONFIGDIR"] = str(Path(SCRATCH) / ".mplconfig")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))
from src.analysis.rois import identity, max, mean, median, roi_dataframes, std
from src.eigenimage.compute_batch import T_LENGTH

DATA = Path(__file__).resolve().parent.parent.parent / "data"
NIIS = DATA / "niis"
ROIS = DATA / "rois"
if not ROIS.exists():
    os.makedirs(ROIS, exist_ok=True)
    os.makedirs(ROIS / "ctrl", exist_ok=True)
    os.makedirs(ROIS / "autism", exist_ok=True)
EIGS = DATA / "eigs"  # for normalizing
SUBJ_DATA = DATA / "Phenotypic_V1_0b_preprocessed1.csv"
EIGIMGS = DATA / "eigimgs"
ATLAS_DIR = DATA / "atlases"
ATLAS = ATLAS_DIR / "cc400_roi_atlas_ALIGNED.nii.gz"
LEGEND = ATLAS_DIR / "CC400_ROI_labels.csv"


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
    weight_sharing: Literal["brains", "rois", "voxels"] = "rois",
) -> DataFrame:
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
    if weight_sharing != "rois":
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
    scores = DataFrame(index=names, columns=["acc"])
    for roi, name in tqdm(enumerate((names)), total=len(names), desc="Fitting SVM for ROI"):
        X = np.stack(df.iloc[:, roi].to_numpy())
        X = StandardScaler().fit_transform(X)
        y = df["target"].to_numpy()
        res = cross_val_score(SVC(), X, y, cv=5, scoring="accuracy")
        scores.loc[name, "acc"] = np.mean(res)
    return scores, guess


if __name__ == "__main__":
    scores, guess = predict_from_roi_reductions(
        source="func",
        norm=None,
        reducer=std,
        slice_reducer=identity,
    )
    print(f"Mean acc: {np.round(np.mean(scores), 3).item()}  (guess = {np.round(guess, 3)})")
    print(
        f"CI: ({np.round(np.percentile(scores, 5), 3)}, {np.round(np.percentile(scores, 95), 3)})"
    )
