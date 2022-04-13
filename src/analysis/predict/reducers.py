import os
import sys
from argparse import Namespace
from dataclasses import dataclass
from pathlib import Path
from pprint import pprint
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import nibabel as nib
import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
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
from src.eigenimage.compute import eigs_via_transpose
from src.eigenimage.compute_batch import T_LENGTH

DATA = Path(__file__).resolve().parent.parent.parent.parent / "data"
SUBJ_DATA = DATA / "Phenotypic_V1_0b_preprocessed1.csv"


@dataclass
class RoiReduction:
    """Reduce each ROI to a single sequence (possibly of length 1) via `reducer`

    Parameters
    ----------
    source: Literal["func", "eigimg"]
        Whether we are reducing the original functional or eigenimages
    nii: Path
        Path to eigimg.nii.gz file
    label: int
        int class label
    legend: DataFrame
        legend from `parse_legend`
    eigens: path
        path to full eigenvalues for normalization
    norm: literal["div", "diff"]
        if "div", divide by the full eigenvalues. this places all vales in [0, 1].
        if "diff", subtract the full eigenvalues.
    reducer: callable[[ndarray], ndarray] = none
        reducing function. if none, uses np.mean(axis=0). must operate on an array
        of shape (roi_voxels, t) and produce a vector of shape (l,).
    reducer_name: str = none
        label which determines output directory of reducer. if `none`, will try
        to use `reducer.__name__`.
    """

    nii: Path
    label: int
    legend: DataFrame
    eigens: Path
    source: Literal["func", "eigimg"]
    norm: Optional[Literal["div", "diff"]] = "div"
    reducer: Optional[Callable[[ndarray], ndarray]] = None
    reducer_name: Optional[str] = None


@dataclass
class SequenceReduction:
    """Reduce each ROI to a single sequence (possibly of length 1) via `reducer`

    Parameters
    ----------
    source: Literal["func", "eigimg"]
        Whether we are reducing the original functional or eigenimages
    nii: Path
        Path to eigimg.nii.gz file
    label: int
        int class label
    legend: DataFrame
        legend from `parse_legend`
    eigens: path
        path to full eigenvalues for normalization
    norm: literal["div", "diff"]
        if "div", divide by the full eigenvalues. this places all vales in [0, 1].
        if "diff", subtract the full eigenvalues.
    reducer: callable[[ndarray], ndarray] = none
        reducing function. if none, uses np.mean(axis=0). must operate on an array
        of shape (roi_voxels, t) and produce a vector of shape (l,).
    reducer_name: str = none
        label which determines output directory of reducer. if `none`, will try
        to use `reducer.__name__`.
    """

    nii: Path
    source: Literal["func", "eigimg"]
    eigens: Optional[None] = None
    norm: Optional[Literal["div", "diff"]] = "div"
    reducer: Optional[Callable[[ndarray], ndarray]] = None
    reducer_name: Optional[str] = None


def identity(x: ndarray) -> ndarray:
    return cast(ndarray, x)


def mean(x: ndarray) -> ndarray:
    return cast(ndarray, np.mean(x, axis=0))


def median(x: ndarray) -> ndarray:
    return cast(ndarray, np.median(x, axis=0))


def std(x: ndarray) -> ndarray:
    return cast(ndarray, np.std(x, ddof=1, axis=0))


def max(x: ndarray) -> ndarray:
    return cast(ndarray, np.max(x, axis=0))


def pca(x: ndarray) -> ndarray:
    if x.ndim != 2:
        raise ValueError("Can only use PCA on 2D array. Use mask to extract voxels.")
    if x.shape[0] != T_LENGTH - 1:
        x = x.T
    if x.shape[0] != T_LENGTH - 1:
        raise ValueError("Img has unmatched shape.")
    return PCA(n_components=1).fit_transform(x).ravel()  # type: ignore


def alpha(x: ndarray) -> ndarray:
    """Fit an exponential  to the eigenvalues above a natural threshold (e.g. Otsu, Yen)
    and return the fit paramater(s)."""


def eigvals(x: ndarray) -> ndarray:
    eigs: ndarray = eigs_via_transpose(x, covariance=True)
    return eigs[1:]  # type: ignore


def trim(raw: ndarray, source: Literal["func", "eigimg"]) -> Optional[ndarray]:
    if source == "func":
        if raw.shape[-1] < T_LENGTH:
            return None
        else:
            return cast(ndarray, raw[:, :, :, -(T_LENGTH - 1) :])
    else:
        if raw.shape[-1] != (T_LENGTH - 1):  # 175
            return None
        else:
            return raw


def normalize(raw: ndarray, args: Union[SequenceReduction, RoiReduction]) -> ndarray:
    norm, source = args.norm, args.source
    if source == "eigimg" and norm is not None:
        if args.eigens is None:
            raise ValueError("Must pass in full eigenvalues if normalizing eigenimage.")
    if norm == "div":
        if source == "eigimg":
            eigs = np.load(args.eigens)
            img = eigs / raw
        else:
            mean_signal = np.mean(raw, axis=(0, 1, 2))
            img = raw / mean_signal
    elif norm == "diff":
        if source == "eigimg":
            eigs = np.load(args.eigens)
            img = raw - eigs
        else:
            mean_signal = np.mean(raw, axis=(0, 1, 2))
            img = raw - mean_signal
    else:
        norm = None
        img = raw
    return cast(ndarray, img)


def subject_labels(imgs: List[Path]) -> List[int]:
    """Returns a list of labels (0 for ctrl, 1 for autism) given a list of nii paths"""
    subjects = pd.read_csv(SUBJ_DATA, usecols=["FILE_ID", "DX_GROUP"])
    # convert from stupid (1,2)=(AUTISM,CTRL) to (0, 1)=(CTRL, AUTISM)
    subjects["DX_GROUP"] = 2 - subjects["DX_GROUP"]
    subjects.rename(columns={"DX_GROUP": "label", "FILE_ID": "fid"}, inplace=True)
    subjects.index = subjects.fid.to_list()
    subjects.drop(columns="fid")
    fids = [img.stem[: img.stem.find("_func")] for img in imgs]
    labels: List[int] = subjects.loc[fids].label.to_list()
    return labels
