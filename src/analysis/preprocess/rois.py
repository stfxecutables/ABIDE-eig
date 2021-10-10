import os
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Optional, Tuple

import nibabel as nib
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series
from scipy.stats import mannwhitneyu, ttest_ind
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

if os.environ.get("CC_CLUSTER") is not None:
    SCRATCH = os.environ["SCRATCH"]
    os.environ["MPLCONFIGDIR"] = str(Path(SCRATCH) / ".mplconfig")
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
from src.analysis.predict.reducers import (
    RoiReduction,
    identity,
    max,
    mean,
    median,
    normalize,
    pca,
    std,
    subject_labels,
    trim,
)
from src.eigenimage.compute_batch import T_LENGTH

DATA = ROOT / "data"
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

ATLAS_400 = ATLAS_DIR / "cc400_roi_atlas_ALIGNED.nii.gz"
LEGEND_400 = ATLAS_DIR / "CC400_ROI_labels.csv"
ATLAS_200 = ATLAS_DIR / "cc200_roi_atlas_ALIGNED.nii.gz"
LEGEND_200 = ATLAS_DIR / "CC200_ROI_labels.csv"


def parse_legend(legend: Path) -> DataFrame:
    """Get a simple, usable legend with columns "ID" for ROI number, and "Name"
    for human-readable label for each ROI number.
    """
    leg = pd.read_csv(legend)
    if "CC400" in legend.stem or "CC200" in legend.stem:  # Craddock
        # The Cameron-Craddock atlases are derived from multiple atlases, and so each CC ROI
        # can be described as a composite of parts of the base atlases (Eickhoff-Zilles, HO,
        # Talairach-Tournoux). We just want *some* name, and EZ has the least "None" values
        name = "Eickhoff-Zilles"
        df = leg.loc[:, ["ROI number", name]].copy()
        df.rename(columns={"ROI number": "ID", name: "Name"}, inplace=True)
        df.index = df["ID"]
        df.drop(columns="ID", inplace=True)
    else:
        df = leg
    return df


def compute_subject_roi_reductions(
    args: RoiReduction,
) -> DataFrame:
    """Reduce each ROI to a single sequence (possibly of length 1) via `reducer`

    Parameters
    ----------
    args: RoiReduction
        args.nii: Path
            Path to eigimg.nii.gz file
        args.label: int
            int class label
        args.legend: DataFrame
            legend from `parse_legend`
        args.eigens: Path
            Path to full eigenvalues for normalization
        args.norm: Literal["div", "diff"]
            If "div", divide by the full eigenvalues. This places all vales in [0, 1].
            If "diff", subtract the full eigenvalues.
        args.reducer: Callable[[ndarray], ndarray] = None
            Reducing function. If None, uses np.mean(axis=0). Must operate on an array
            of shape (ROI_voxels, T) and produce a vector of shape (L,).
        args.reducer_name: str = None
            Label which determines output directory of reducer. If `None`, will try
            to use `reducer.__name__`.

    Returns
    -------
    reduced: DataFrame
        Where rows are ROIs, columns are:

            "name":     (ROI name),
            "n_voxels": (how many in roi),
            "signal":   (mean signal across voxels)
            "class":    (0=CTRL, 1=AUTISM)
    """
    np.warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)  # type: ignore
    nii, label, legend = args.nii, args.label, args.legend
    reducer, reducer_name = args.reducer, args.reducer_name
    source, norm = args.source, args.norm
    if reducer is None:
        reducer = mean

    raw = nib.load(str(nii)).get_fdata()
    raw = trim(raw, source)
    if raw is None:
        return
    img = normalize(raw, args)
    atlas = nib.load(str(ATLAS)).get_data()
    df = DataFrame(index=legend.index, columns=["name", "n_voxels", "signal"])
    for id in zip(legend.index):
        mask = atlas == id
        roi = img[mask, :]
        n_voxels = roi.shape[0]
        reduction = reducer(roi)
        name = legend.loc[id].item()
        df.loc[id, :] = (name, n_voxels, reduction)
    extensions = "".join(nii.suffixes)
    subdir = "ctrl" if label == 0 else "autism"
    rname = reducer.__name__ if reducer_name is None else reducer_name
    outdir = ROIS / f"{source}/{rname}/{subdir}"
    if not outdir.exists():
        os.makedirs(outdir, exist_ok=True)
    outfile = outdir / nii.name.replace(extensions, ".parquet").replace(
        "_func_minimal", f"_ROI_{rname}_norm={norm}"
    )
    df.to_parquet(str(outfile), index=True)


def compute_all_subject_roi_reductions(
    source: Literal["func", "eigimg"] = "eigimg",
    norm: Optional[Literal["div", "diff"]] = "div",
    reducer: Callable[[ndarray], ndarray] = None,
    reducer_name: str = None,
) -> None:
    if source == "eigimg":
        niis = sorted(EIGIMGS.rglob("*eigimg.nii.gz"))
    else:
        niis = sorted(NIIS.rglob("*func_minimal.nii.gz"))
    eigs = [EIGS / nii.name.replace("_eigimg.nii.gz", ".npy") for nii in niis]
    labels = subject_labels(niis)
    legend = parse_legend(LEGEND)
    args = [
        RoiReduction(
            source=source,
            nii=nii,
            label=label,
            legend=legend,
            eigens=eig,
            norm=norm,
            reducer=reducer,
            reducer_name=reducer_name,
        )
        for nii, eig, label in zip(niis, eigs, labels)
    ]
    # for arg in args:
    #     compute_subject_roi_reductions(args)
    process_map(compute_subject_roi_reductions, args)


def precompute_all_func_roi_reductions() -> None:
    for reducer in [mean, median, max, std, pca]:
        for norm in ["div", "diff", None]:
            try:
                compute_all_subject_roi_reductions(
                    source="func",
                    norm=norm,  # type: ignore
                    reducer=reducer,
                )
            except Exception as e:
                print(f"Got exception {e}")
                traceback.print_exc()


def precompute_all_eigimg_roi_reductions() -> None:
    for reducer in [mean, median, max, std, pca]:
        for norm in ["div", "diff", None]:
            try:
                compute_all_subject_roi_reductions(
                    source="eigimg",
                    norm=norm,  # type: ignore
                    reducer=reducer,
                )
            except Exception as e:
                print(f"Got exception {e}")
                traceback.print_exc()


def eig_descriptives() -> DataFrame:
    niis = sorted(EIGIMGS.rglob("*eigimg.nii.gz"))
    eigs = [np.load(EIGS / nii.name.replace("_eigimg.nii.gz", ".npy")) for nii in niis]
    labels = subject_labels(niis)
    autism, ctrl = [], []
    eig: ndarray
    dfs = []
    for idx in range(len(eigs[0]) - 1):
        for eig, label in zip(eigs, labels):
            if len(eig) != 175:
                continue
            value = eig[idx]
            # value = np.percentile(eig, [89])
            if label == 1:
                autism.append(value)
            else:
                ctrl.append(value)
        aut, ctr = np.array(autism), np.array(ctrl)
        t, t_p = ttest_ind(ctr, aut, equal_var=False)
        U, U_p = mannwhitneyu(ctr, aut, alternative="two-sided")
        d = cohens_d(ctr, aut)
        ac = auc(ctr, aut)
        df = DataFrame(
            data=[(t, t_p, U, U_p, d, ac)],
            columns=["t", "t_p", "U", "U_p", "d", "AUC"],
            index=[idx],
        )
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    return df


def cohens_d(x1: DataFrame, x2: DataFrame) -> float:
    n1, n2 = len(x1) - 1, len(x2) - 1
    sd_pool = (n1 * np.std(x1, ddof=1) + n2 * np.std(x2, ddof=1)) / (n1 + n2)
    return float((np.mean(x2) - np.mean(x1)) / sd_pool)


def auc(x1: DataFrame, x2: DataFrame) -> float:
    y_true = np.concatenate([np.zeros(len(x1)), np.ones(len(x2))])
    y_score = np.concatenate([np.array(x1).ravel(), np.array(x2).ravel()])
    return float(roc_auc_score(y_true, y_score))


def roi_dataframes(
    source: Literal["func", "eigimg"] = "eigimg",
    norm: Optional[Literal["div", "diff"]] = "div",
    reducer: Callable[[ndarray], ndarray] = None,
    reducer_name: str = None,
    slicer: slice = slice(None),
    slice_reducer: Callable[[ndarray], ndarray] = mean,
) -> Tuple[DataFrame, DataFrame, Series]:
    if reducer is None:
        reducer = mean
    rname = reducer.__name__ if reducer_name is None else reducer_name
    regex = f"*_ROI_{rname}_norm={norm}{'_eigimg' if source == 'eigimg' else ''}.parquet"
    autism_dir = ROIS / f"{source}/{rname}/autism"
    ctrl_dir = ROIS / f"{source}/{rname}/ctrl"
    autisms = [pd.read_parquet(p) for p in sorted(autism_dir.rglob(regex))]
    ctrls = [pd.read_parquet(p) for p in sorted(ctrl_dir.rglob(regex))]
    names = autisms[0]["name"].copy()

    for df in autisms:
        df.signal = df.signal.apply(lambda s: slice_reducer(s[slicer]))
        df.drop(columns=["name", "n_voxels"], inplace=True)
    for df in ctrls:
        df.signal = df.signal.apply(lambda s: slice_reducer(s[slicer]))
        df.drop(columns=["name", "n_voxels"], inplace=True)
    autism = pd.concat(autisms, axis=1)
    ctrl = pd.concat(ctrls, axis=1)
    return autism, ctrl, names


def compute_roi_descriptive_stats(
    source: Literal["func", "eigimg"] = "eigimg",
    norm: Literal["div", "diff"] = "div",
    reducer: Callable[[ndarray], ndarray] = None,
    reducer_name: str = None,
    slicer: slice = slice(None),
    slice_reducer: Callable[[ndarray], ndarray] = mean,
) -> DataFrame:
    autism, ctrl, names = roi_dataframes(
        source=source,
        norm=norm,
        reducer=reducer,
        reducer_name=reducer_name,
        slicer=slicer,
        slice_reducer=slice_reducer,
    )
    # autism.index, ctrl.index = names, names
    print(f"Got {autism.shape[1]} autism and {ctrl.shape[1]} ctrl subjects.")
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
    return descriptives
    # print(f"Smallest differences")
    # print(
    #     descriptives.sort_values(by=["U_p", "d"], ascending=True)
    #     .iloc[-10:, :]
    #     .to_markdown(tablefmt="simple", floatfmt="1.3f")
    # )


def print_descriptives(df: DataFrame) -> None:
    print(
        df.sort_values(by=["U_p", "t_p"], ascending=True)
        .iloc[:20, :]
        .to_markdown(
            tablefmt="simple", floatfmt=["0.1f", "1.2f", "1.1e", "1.0f", "1.1e", "1.3f", "1.3f"]
        )
    )


if __name__ == "__main__":
    for norm in [None, "div"]:
        compute_all_subject_roi_reductions(
            source="func",
            norm=norm,
            reducer=pca,
        )
    sys.exit()
    print(
        eig_descriptives()
        .sort_values(by="U_p", ascending=True)
        .to_markdown(
            tablefmt="simple", floatfmt=["1.2f", "1.2f", "1.1e", "1.2f", "1.1e", "1.3f", "1.3f"]
        )
    )
    print_descriptives(
        compute_roi_descriptive_stats(
            source="func",
            norm=None,
            reducer=std,
            # slicer=slice(100, 127),
            slice_reducer=mean,
        )
    )
    sys.exit()
    dfs: List[DataFrame] = []
    for i in range(1, 30):
        df = compute_roi_descriptive_stats(
            source="eigimg", norm="div", reducer=std, slicer=slice(-i)
        )
        df["idx"] = -i
        dfs.append(df)
    df = pd.concat(dfs, axis=0)
    print_descriptives(df)
