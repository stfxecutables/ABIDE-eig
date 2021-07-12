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
import sys
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type, cast

import nibabel as nib
import numpy as np
import optuna
import pandas as pd
from numpy import ndarray
from sklearn.svm import SVC
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from src.analysis.predict.hypertune import HtuneResult, hypertune_classifier
from src.analysis.predict.reducers import (
    SequenceReduction,
    eigvals,
    identity,
    mean,
    normalize,
    subject_labels,
    trim,
)

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
LOG = os.environ.get("CC_CLUSTER") is None
VERBOSITY = optuna.logging.INFO if LOG else optuna.logging.ERROR


def compute_sequence_reduction(
    args: SequenceReduction,
) -> Optional[ndarray]:
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
        return cast(ndarray, np.load(outfile))

    if reducer is None:
        raise ValueError("Must have some reducer for seqeuence reduction.")
    raw = nib.load(str(nii)).get_fdata()
    # trim
    raw = trim(raw, source)
    if raw is None:
        return None
    img = normalize(raw, args)
    mask = nib.load(str(MASK)).get_fdata().astype(bool)
    voxels = img[mask]

    result = reducer(voxels)
    np.save(outfile, result)
    return cast(ndarray, result)


def compute_sequence_reductions(
    source: Literal["func", "eigimg"] = "eigimg",
    norm: Optional[Literal["div", "diff"]] = "div",
    reducer: Callable[[ndarray], ndarray] = mean,
    reducer_name: str = None,
) -> None:
    NII_DIR = NIIS if source == "func" else EIGIMGS
    NII_REGEX = "*func_minimal.nii.gz" if source == "func" else "*eigimg.nii.gz"
    niis = sorted(NII_DIR.rglob(NII_REGEX))
    eigs = [
        Path(str(nii).replace("eigimgs", "eigs").replace("_eigimg.nii.gz", ".npy")) for nii in niis
    ]
    args = [
        SequenceReduction(
            nii=nii,
            source=source,
            eigens=eig if source == "eigimg" else None,
            norm=norm,
            reducer=reducer,
            reducer_name=reducer_name,
        )
        for nii, eig in zip(niis, eigs)
    ]
    results = process_map(compute_sequence_reduction, args)
    all_labels = subject_labels(niis)
    if len(results) == 0:
        logging.debug("No sequence reductions to concatenate into DataFrame. All done.")
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
    logging.debug(f"Saved {reducer_name} spectrum reduction to {outfile}.")


def predict_from_sequence_reductions(
    source: Literal["func", "eigimg"] = "eigimg",
    norm: Literal["div", "diff"] = "div",
    reducer: Callable[[ndarray], ndarray] = mean,
    reducer_name: str = None,
    slicer: slice = slice(None),
    slice_reducer: Callable[[ndarray], ndarray] = identity,
    classifier: Type = SVC,
    classifier_args: Dict[str, Any] = {},
) -> Tuple[ndarray, HtuneResult]:
    if source == "eigimg":
        reducer = eigvals
        reducer_name = reducer.__name__

    rname = reducer.__name__ if reducer_name is None else reducer_name
    outdir = SEQS / f"{source}/{rname}"
    df_path = outdir / f"SEQ_ALL_{rname}_norm={norm}.parquet"
    df = pd.read_parquet(df_path)
    X = df.drop(columns="target").to_numpy()
    y = df["target"].to_numpy().ravel()
    guess = np.max([np.sum(y == 0), np.sum(y == 1)]) / len(y)
    htune_result = hypertune_classifier("rf", X, y, n_trials=500, cv_method=5, verbosity=VERBOSITY)
    # res = cross_val_score(classifier(**classifier_args), X, y, cv=5, scoring="accuracy")
    return guess, htune_result
