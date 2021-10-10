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
from numpy.core.fromnumeric import trace
from pandas import DataFrame
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

CC_CLUSTER = os.environ.get("CC_CLUSTER")
if CC_CLUSTER is not None:
    SCRATCH = os.environ["SCRATCH"]
    os.environ["MPLCONFIGDIR"] = str(Path(SCRATCH) / ".mplconfig")
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
from src.eigenimage.compute import eigs_via_transpose

"""
Notes
-----

We need to extract:

- roi_means:
- roi_sds:
- r_mean: (ROI mean correlations)
- r_sd: (ROI sd correlations)
- lap_mean: (T-thresholded Laplacian eigenvalues of r_mean)
- lap_sd: (T-thresholded Laplacian eigenvalues of r_sd)
- eig_mean: (Eigenvalues of r_mean)
- eig_sd: (Eigenvalues of r_sd)
- eig_full: (Eigenvalues of all brain voxels)
- eig_full_p: (eig_full padding short scans prior to computing eigs)
- eig_full_c: (eig_full cropping long scans prior to computing eigs)
- eig_full_pc: (eig_full padding short and cropping long scans prior to computing eigs)
"""

SUFFIX = "_f16_subsample" if CC_CLUSTER is None else ""
DATA = ROOT / "data"
NIIS = DATA / f"nii_cpac{SUFFIX}"
FEATURES_DIR = DATA / f"features_cpac{SUFFIX}"


ROIS = DATA / "rois"
if not ROIS.exists():
    os.makedirs(ROIS, exist_ok=True)
    os.makedirs(ROIS / "ctrl", exist_ok=True)
    os.makedirs(ROIS / "autism", exist_ok=True)
EIGS = DATA / "eigs"  # for normalizing
SUBJ_DATA = DATA / "Phenotypic_V1_0b_preprocessed1.csv"
EIGIMGS = DATA / "eigimgs"

ATLAS_DIR = DATA / "atlases"
ATLAS_400 = ATLAS_DIR / "cc400_roi_atlas_ALIGNED.nii.gz"
LEGEND_400 = ATLAS_DIR / "CC400_ROI_labels.csv"
ATLAS_200 = ATLAS_DIR / "cc200_roi_atlas_ALIGNED.nii.gz"
LEGEND_200 = ATLAS_DIR / "CC200_ROI_labels.csv"
ATLASES = {"cc200": ATLAS_200, "cc400": ATLAS_400}
LEGENDS = {"cc200": LEGEND_200, "cc400": LEGEND_400}
MASK = ATLAS_DIR / "MASK.nii.gz"


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


def compute_roi_descriptives(
    arr: ndarray, atlas: ndarray, summary: Literal["mean", "sd"]
) -> ndarray:
    """Return array of shape (T, n_ROI) where ROIs are summarized via statistic in `summary`"""
    roi_ids = np.unique(atlas)[1:]  # exclude 0 == air voxels
    rois = np.empty((arr.shape[-1], len(roi_ids)))
    for i, rid in enumerate(roi_ids):
        roi = arr[atlas == rid]
        seq = np.mean(roi) if summary == "mean" else np.std(roi, ddof=1)
        rois[:, i] = seq
    return rois


def compute_desc_correlations(desc: ndarray) -> ndarray:
    """Assumes `desc` has shape (T, n_ROI).

    Returns
    -------
    corrs: ndarray
        Array of Pearson correlations of ROIs, shape (n_ROI, n_ROI)
    """
    return np.corrcoef(desc, rowvar=False)


def compute_corr_eigs(corrs: ndarray) -> ndarray:
    """Compute the eigenvalues of the correlation matrix"""
    return np.linalg.eigvalsh(corrs)


def compute_full_eigs(arr: ndarray, T: int = 203, crop: bool = False, pad: bool = False) -> ndarray:
    """Compute the full eigenvalues using transpose trick.

    Parameters
    ----------
    arr: ndarray
        4D fMRI data.

    T: int = 203
        Length to pad or crop to. Ignored if `pad` and `crop` are both False. We use 203 because
        a correlation matrix where the shortest dimension is size T has T - 1 eigenvalues, and
        MinMax normalization sets the smallest and largest eigenvalues to 0 and 1 respectively for
        all matrices, i.e. we lose 3 eigenvalues total.

    pad: bool = False
        If True, use np.pad(arr, (0, T), "wrap") for short scans.
    crop: bool = False
        If True, take a slice of size T (last timepoints cropped away).
    """
    mask = nib.load(str(MASK)).get_fdata().astype(bool)
    brain = arr[mask, :]
    # remove constant voxels, which result in undefined sd, correlation
    constants = brain.std(axis=1) <= 1e-15  # almost np.finfo(np.float64).eps
    brain = brain[~constants]
    # resize
    t = brain.shape[-1]
    if crop and t > T:
        brain = brain[:, :T]
    if pad and t < T:
        brain = np.pad(brain, ((0, T - t), (0, 0)), "wrap")
    eigs = eigs_via_transpose(brain, covariance=False)
    return eigs[1:]


def compute_laplacian_eigs(corrs: ndarray) -> Tuple[ndarray, ndarray]:
    """
    Notes
    -----
    We directly follow the procedure in Section II. D of

        Mostafa, S., Tang, L., & Wu, F.-X. (2019). Diagnosis of Autism Spectrum Disorder Based on
        Eigenvalues of Brain Networks. IEEE Access, 7, 128474–128486.
        https://doi.org/10.1109/ACCESS.2019.2940198

    Note however the procedure is a bit dubious, as when dealing with Laplacians of directed graphs
    we should have two degree matrices, depending on whether we are counting indegrees or
    outdegrees. Presumably, the authors are treating here the matrix as a *weighted* directed graph,
    in order to ignore this issue, but then it is not clear why a threshold T is used in the first
    place...

    As per the above paper, and also:

        Yin, W., Mostafa, S., & Wu, F. (2021). Diagnosis of Autism Spectrum Disorder Based on
        Functional Brain Networks with Deep Learning. Journal of Computational Biology, 28(2),
        146–165. https://doi.org/10.1089/cmb.2020.0252

    the values T = 0.2 and T = 0.4 produce highest accuracies for the ABIDE data, so we just just
    those two.
    """
    # construct adjacency matrix A using threshold T
    eigs = []
    for T in [0.2, 0.4]:
        A = np.copy(corrs)
        A[np.abs(corrs) < T] = 0
        A[corrs >= T] = 1
        A[corrs <= -T] = -1
        A[np.diag_indices_from(A)] = 0
        D = np.diag(np.sum(A, axis=1))  # construct degree matrix D
        L = D - A  # Laplacian
        eigs.append(np.linalg.eigvalsh(L))
    return eigs[0], eigs[1]


def extract_features(nii: Path) -> None:
    try:
        arr = nib.load(str(nii)).get_fdata()
        for atlas_name, atlas in ATLASES.items():
            # even with CC400 matrix each result is only 400x400, e.g. 1.2 MB max
            roi_means = compute_roi_descriptives(arr, atlas, "mean")
            roi_sds = compute_roi_descriptives(arr, atlas, "sd")
            r_mean = compute_desc_correlations(roi_means)
            r_sd = compute_desc_correlations(roi_sds)
            lap_mean02, lap_mean04 = compute_laplacian_eigs(r_mean)
            lap_sd02, lap_sd04 = compute_laplacian_eigs(r_sd)
            eig_mean = compute_corr_eigs(r_mean)
            eig_sd = compute_corr_eigs(r_sd)
            eig_full = compute_full_eigs(arr, T=203, crop=False, pad=False)
            eig_full_p = compute_full_eigs(arr, T=203, crop=False, pad=True)
            eig_full_c = compute_full_eigs(arr, T=203, crop=True, pad=False)
            eig_full_pc = compute_full_eigs(arr, T=203, crop=True, pad=True)
            features = dict(
                roi_means=roi_means,
                roi_sds=roi_sds,
                r_mean=r_mean,
                r_sd=r_sd,
                lap_mean02=lap_mean02,
                lap_mean04=lap_mean04,
                lap_sd02=lap_sd02,
                lap_sd04=lap_sd02,
                eig_mean=eig_mean,
                eig_sd=eig_sd,
                eig_full=eig_full,
                eig_full_p=eig_full_p,
                eig_full_c=eig_full_c,
                eig_full_pc=eig_full_pc,
            )
            atlas_outdir = FEATURES_DIR / atlas_name
            for feature_name, feature in features.items():
                outdir = atlas_outdir / feature_name
                outfile = outdir / nii.name.replace("func_preproc.nii.gz", f"_{feature_name}.npy")
                if not outdir.exists():
                    os.makedirs(outdir, exist_ok=True)
                np.save(outfile, feature, allow_pickle=False, fix_imports=False)
    except Exception as e:
        traceback.print_exc()
        print(f"Got error for subject {nii}:")
        print(e)


if __name__ == "__main__":
    niis = sorted(NIIS.rglob("*.nii.gz"))[:30]
    process_map(extract_features, niis, desc="Extracting features")
