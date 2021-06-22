import os
import sys
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
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
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

    source: Literal["func", "eigimg"]
    nii: Path
    label: int
    legend: DataFrame
    eigens: Path
    norm: Optional[Literal["div", "diff"]] = "div"
    reducer: Optional[Callable[[ndarray], ndarray]] = None
    reducer_name: Optional[str] = None


def mean(x: ndarray) -> ndarray:
    return np.mean(x, axis=0)


def median(x: ndarray) -> ndarray:
    return np.median(x, axis=0)


def std(x: ndarray) -> ndarray:
    return np.std(x, ddof=1, axis=0)


def max(x: ndarray) -> ndarray:
    return np.max(x, axis=0)


def pca(x: ndarray) -> ndarray:
    pc = PCA(1, whiten=False)
    return pc.fit_transform(x.T)


def identity(x: ndarray) -> ndarray:
    return x


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
    nii, label, legend, eigens = args.nii, args.label, args.legend, args.eigens
    reducer, reducer_name = args.reducer, args.reducer_name
    source, norm = args.source, args.norm
    if reducer is None:
        reducer = mean

    raw = nib.load(str(nii)).get_fdata()
    # trim
    if source == "func":
        if raw.shape[-1] < T_LENGTH:
            return
        elif raw.shape[-1] >= T_LENGTH:
            raw = raw[:, :, :, -(T_LENGTH - 1) :]
    else:
        if raw.shape[-1] != (T_LENGTH - 1):
            return
    # normalize
    if norm == "div":
        if source == "eigimg":
            eigs = np.load(eigens)
            img = eigs / raw
        else:
            mean_signal = np.mean(raw, axis=(0, 1, 2))
            img = raw / mean_signal
    elif norm == "diff":
        if source == "eigimg":
            eigs = np.load(eigens)
            img = raw - eigs
        else:
            mean_signal = np.mean(raw, axis=(0, 1, 2))
            img = raw - mean_signal
    else:
        norm = None
        img = raw

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
    for reducer in [mean, median, max, std]:
        for norm in ["div", "diff", None]:
            compute_all_subject_roi_reductions(
                source="func",
                norm=norm,  # type: ignore
                reducer=reducer,
            )


def precompute_all_eigimg_roi_reductions() -> None:
    for reducer in [mean, median, max, std]:
        for norm in ["div", "diff", None]:
            compute_all_subject_roi_reductions(
                source="eigimg",
                norm=norm,  # type: ignore
                reducer=reducer,
            )


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
