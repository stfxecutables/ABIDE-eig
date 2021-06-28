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
from src.analysis.predict.hypertune import hypertune_classifier
from src.analysis.predict.reducers import SequenceReduction, normalize, subject_labels, trim
from src.analysis.rois import identity, max, mean, median, pca, roi_dataframes, std
from src.eigenimage.compute_batch import T_LENGTH

DATA = Path(__file__).resolve().parent.parent.parent / "data"
NIIS = DATA / "niis"
SEQS = DATA / "seqs"
if not SEQS.exists():
    os.makedirs(SEQS, exist_ok=True)
    os.makedirs(SEQS / "ctrl", exist_ok=True)
    os.makedirs(SEQS / "autism", exist_ok=True)
EIGS = DATA / "eigs"  # for normalizing
SUBJ_DATA = DATA / "Phenotypic_V1_0b_preprocessed1.csv"
EIGIMGS = DATA / "eigimgs"
ATLAS_DIR = DATA / "atlases"
MASK = ATLAS_DIR / "MASK.nii.gz"
ATLAS = ATLAS_DIR / "cc400_roi_atlas_ALIGNED.nii.gz"
LEGEND = ATLAS_DIR / "CC400_ROI_labels.csv"


def compute_sequence_reduction(
    args: SequenceReduction,
) -> ndarray:
    source, nii = args.source, args.nii
    reducer = args.reducer
    if reducer is None:
        raise ValueError("Must have some reducer for seqeuence reduction.")
    raw = nib.load(str(nii)).get_fdata()
    # trim
    raw = trim(raw, source)
    if raw is None:
        return
    img = normalize(raw, args)
    mask = nib.load(str(MASK)).get_fdata().astype(bool)
    voxels = img[mask]
    return reducer(voxels)


def compute_sequence_reductions(
    source: Literal["func", "eigimg"] = "eigimg",
    norm: Optional[Literal["div", "diff"]] = "div",
    reducer: Callable[[ndarray], ndarray] = identity,
    reducer_name: str = None,
) -> None:
    if source == "eigimg":
        return
    niis = sorted(NIIS.rglob("*func_minimal.nii.gz"))
    args = [
        SequenceReduction(
            nii=nii, source=source, norm=norm, reducer=reducer, reducer_name=reducer_name
        )
        for nii in niis
    ]
    results = process_map(compute_sequence_reduction, args)
    all_labels = subject_labels(niis)
    if len(results) == 0:
        print("No sequence reductions to concatenate into DataFrame. All done.")
        return
    reductions, labels = [], []
    for result, label in zip(results, all_labels):
        if result is not None:
            reductions.append(result)
            labels.append(label)
    df = pd.DataFrame(data=np.vstack(reductions))
    df["target"] = labels

    rname = reducer.__name__ if reducer_name is None else reducer_name
    outdir = SEQS / f"{source}/{rname}"
    if not outdir.exists():
        os.makedirs(outdir, exist_ok=True)
    outfile = outdir / f"SEQ_{rname}_norm={norm}"
    df.to_parquet(str(outfile), index=True)
    print(f"Saved {reducer_name} spectrum reduction to {outfile}.")


def predict_from_sequence_reductions(
    source: Literal["func", "eigimg"] = "eigimg",
    norm: Literal["div", "diff"] = "div",
    reducer: Callable[[ndarray], ndarray] = None,
    reducer_name: str = None,
    slicer: slice = slice(None),
    slice_reducer: Callable[[ndarray], ndarray] = identity,
    classifier: Type = SVC,
    classifier_args: Dict[str, Any] = {},
) -> None:
    pass
