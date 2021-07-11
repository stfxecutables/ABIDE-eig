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
from src.analysis.predict.hypertune import HtuneResult, hypertune_classifier
from src.analysis.predict.reducers import (
    SequenceReduction,
    eigvals,
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
from src.analysis.rois import roi_dataframes
from src.eigenimage.compute_batch import T_LENGTH

DATA = Path(__file__).resolve().parent.parent.parent.parent / "data"
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
    reducer, reducer_name = args.reducer, args.reducer_name
    norm = args.norm

    # early return if already done
    rname = reducer.__name__ if reducer_name is None else reducer_name
    outdir = SEQS / f"{source}/{rname}"
    if not outdir.exists():
        os.makedirs(outdir, exist_ok=True)
    fname = Path(str(nii).replace(".nii.gz", "")).name
    outfile = outdir / f"SEQ_{rname}_norm={norm}_{fname}.npy"
    if outfile.exists():
        return np.load(outfile)

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

    result = reducer(voxels)
    np.save(outfile, result)
    return result


def compute_sequence_reductions(
    source: Literal["func", "eigimg"] = "eigimg",
    norm: Optional[Literal["div", "diff"]] = "div",
    reducer: Callable[[ndarray], ndarray] = mean,
    reducer_name: str = None,
) -> None:
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
    cols = [str(i) for i in range(len(reductions[0]))]
    df = pd.DataFrame(data=np.vstack(reductions), columns=cols)
    df["target"] = labels

    rname = reducer.__name__ if reducer_name is None else reducer_name
    outdir = SEQS / f"{source}/{rname}"
    if not outdir.exists():
        os.makedirs(outdir, exist_ok=True)
    outfile = outdir / f"SEQ_ALL_{rname}_norm={norm}.parquet"
    df.to_parquet(str(outfile), index=True)
    print(f"Saved {reducer_name} spectrum reduction to {outfile}.")


def predict_from_sequence_reductions(
    source: Literal["func", "eigimg"] = "eigimg",
    norm: Literal["div", "diff"] = "div",
    reducer: Callable[[ndarray], ndarray] = mean,
    reducer_name: str = None,
    slicer: slice = slice(None),
    slice_reducer: Callable[[ndarray], ndarray] = identity,
    classifier: Type = SVC,
    classifier_args: Dict[str, Any] = {},
) -> Tuple[DataFrame, ndarray, HtuneResult]:
    if source == "eigimg":
        reducer = eigvals
        reducer_name = reducer.__name__

    rname = reducer.__name__ if reducer_name is None else reducer_name
    df_path = SEQS / f"{source}/{rname}/SEQ_ALL_{rname}_norm={norm}.parquet"

    df = pd.read_parquet(df_path)
    X = df.drop(columns="target").to_numpy()
    y = df["target"].to_numpy().ravel()
    guess = np.max([np.sum(y == 0), np.sum(y == 1)]) / len(y)
    pprint(classifier_args, indent=2)
    htune_result = hypertune_classifier(
        "rf", X, y, n_trials=200, cv_method=5, verbosity=optuna.logging.INFO
    )
    # res = cross_val_score(classifier(**classifier_args), X, y, cv=5, scoring="accuracy")
    print(f"Best val_acc: {htune_result.val_acc}")
    scores = pd.DataFrame(index=["all"], columns=["acc"], data=htune_result.val_acc)
    return scores, guess, htune_result


if __name__ == "__main__":
    # NOTE: `norm` arg is fucked for eigimgs, needs to be done after too
    scores, guess = predict_from_sequence_reductions(
        source="func",
        norm="diff",
        reducer=std,
        classifier=RandomForestClassifier,
    )
    print(f"Mean acc: {np.round(np.mean(scores), 3).item()}  (guess = {np.round(guess, 3)})")
    print(
        f"CI: ({np.round(np.percentile(scores, 5), 3)}, {np.round(np.percentile(scores, 95), 3)})"
    )

"""
Guess = 0.569
Results: Mean
    Best Acc: 0.591
        source="func", norm="div", reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.556
        source="func", norm="diff", reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.562
        source="func", norm=None, reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,

Results: Std
    Best Acc: 0.551
        source="func", norm="div", reducer=std,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.569
        source="func", norm="diff", reducer=std,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.573
        source="func", norm=None, reducer=std,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,

TODO: test PCA reduced ROIs and/or PCA reduced fMRI images
Results: PCA
    Best Acc: 0.587
        source="func", norm="div", reducer=PCA,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.571
        source="eigimg", norm="diff", reducer=PCA,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.553
        source="eigimg", norm=None, reducer=PCA,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,

Results: Eigimg
    Best Acc: 0.618
        source="func", norm="div", reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.589
        source="func", norm="diff", reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.593
        source="func", norm=None, reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,

"""
