import os
from argparse import Namespace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from numpy.core.numeric import ones_like
from pandas import DataFrame, Series
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

DATA = Path(__file__).resolve().parent.parent.parent / "data"
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


def parse_legend(legend: Path) -> DataFrame:
    """Get a simple, usable legend with columns "ID" for ROI number, and "Name"
    for human-readable label for each ROI number.
    """
    leg = pd.read_csv(legend)
    if "CC400" in legend.stem or "CC200" in legend.stem:  # Craddock
        name = "Talairach-Tournoux"
        df = leg.loc[:, ["ROI number", name]].copy()
        df.rename(columns={"ROI number": "ID", name: "Name"}, inplace=True)
        df.index = df["ID"]
        df.drop(columns="ID", inplace=True)
    else:
        df = leg
    return df


def subject_labels(imgs: List[Path]) -> List[int]:
    subjects = pd.read_csv(SUBJ_DATA, usecols=["FILE_ID", "DX_GROUP"])
    # convert from stupid (1,2)=(AUTISM,CTRL) to (0, 1)=(CTRL, AUTISM)
    subjects["DX_GROUP"] = 2 - subjects["DX_GROUP"]
    subjects.rename(columns={"DX_GROUP": "label", "FILE_ID": "fid"}, inplace=True)
    subjects.index = subjects.fid.to_list()
    subjects.drop(columns="fid")
    fids = [img.stem[: img.stem.find("_func")] for img in imgs]
    labels: List[int] = subjects.loc[fids].label.to_list()
    return labels


def compute_roi_means(args: Namespace) -> DataFrame:
    """return DataFrame where rows are ROIs, columns are:

    "name":     (ROI name),
    "n_voxels": (how many in roi),
    "signal":   (mean signal across voxels)
    "class":    (0=CTRL, 1=AUTISM)
    """
    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
    nii, label, legend, eigens = args.nii, args.label, args.legend, args.eigens
    img = nib.load(str(nii)).get_fdata()
    atlas = nib.load(str(ATLAS)).get_data()
    eigs = np.load(eigens)
    df = DataFrame(index=legend.index, columns=["name", "n_voxels", "signal"])
    for id in zip(legend.index):
        mask = atlas == id
        roi = img[mask, :]
        n_voxels = roi.shape[0]
        mean = np.mean(roi, axis=0)
        name = legend.loc[id].item()
        df.loc[id, :] = (name, n_voxels, mean)
    extensions = "".join(nii.suffixes)
    outdir = ROIS / ("ctrl" if label == 0 else "autism")
    outfile = outdir / nii.name.replace(extensions, ".parquet").replace(
        "_func_minimal", "_ROI_means"
    )
    df.to_parquet(str(outfile), index=True)


def compute_roi_mean_signals() -> None:
    niis = sorted(EIGIMGS.rglob("*eigimg.nii.gz"))
    eigs = [EIGS / nii.name.replace("_eigimg.nii.gz", ".npy") for nii in niis]
    labels = subject_labels(niis)
    legend = parse_legend(LEGEND)
    args = [
        Namespace(**dict(nii=nii, label=label, legend=legend, eigens=eig)) for nii, eig, label in zip(niis, eigs, labels)
    ]
    process_map(compute_roi_means, args)


def cohens_d(x1: DataFrame, x2: DataFrame) -> float:
    n1, n2 = len(x1) - 1, len(x2) - 1
    sd_pool = (n1 * np.std(x1, ddof=1) + n2 * np.std(x2, ddof=1)) / (n1 + n2)
    return (np.mean(x1) - np.mean(x2)) / sd_pool


def auc(x1: DataFrame, x2: DataFrame) -> float:
    y_true = np.concatenate([np.zeros_like(x1), np.ones_like(x2)])
    y_score = np.concatenate([x1, x2])
    return roc_auc_score(y_true, y_score)


def compute_roi_largest_descriptive_stats(n_largest: int = 1) -> DataFrame:
    autisms = [pd.read_parquet(p) for p in sorted((ROIS / "autism").rglob("*.parquet"))]
    ctrls = [pd.read_parquet(p) for p in sorted((ROIS / "ctrl").rglob("*.parquet"))]
    names = autisms[0]["name"].copy()
    n_voxels = autisms[0]["n_voxels"].copy()

    for df in autisms:
        df.signal = df.signal.apply(lambda s: s[:-n_largest].mean())
        df.drop(columns=["name", "n_voxels"], inplace=True)
    for df in ctrls:
        df.signal = df.signal.apply(lambda s: s[:-n_largest].mean())
        df.drop(columns=["name", "n_voxels"], inplace=True)

    autism = pd.concat(autisms, axis=1)
    ctrl = pd.concat(ctrls, axis=1)
    # autism.index, ctrl.index = names, names
    descriptives = DataFrame(index=names, columns=["t", "t_p", "U", "U_p", "d", "AUC"])
    for roi in tqdm(range(len(names)), total=len(names)):
        aut = np.array(autism.iloc[roi, :])
        ctr = np.array(ctrl.iloc[roi, :])
        t, t_p = ttest_ind(ctr, aut, equal_var=False)
        U, U_p = mannwhitneyu(ctr, aut, alternative="two-sided")
        d = cohens_d(ctr, aut)
        ac = auc(ctr, aut)
        descriptives.iloc[roi, :] = (t, t_p, U, U_p, d, ac)
    descriptives.index = names
    print(
        descriptives.sort_values(by="U_p", ascending=True).to_markdown(
            tablefmt="simple", floatfmt="1.3f"
        )
    )


if __name__ == "__main__":
    # compute_roi_mean_signals()
    compute_roi_largest_descriptive_stats(n_largest=5)
