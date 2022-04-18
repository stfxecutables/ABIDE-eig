# fmt: off
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
logging.getLogger("tensorflow").setLevel(logging.FATAL)
logging.getLogger("tensorboard").setLevel(logging.FATAL)
# fmt: on

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from joblib import Memory
from numpy import ndarray
from torch import Tensor
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from replication_experiments.constants import Norm

MEMOIZER = Memory(ROOT / "__CACHE__")


def get_labels_sites(imgs: List[Path], subj_data: Path) -> Tuple[List[int], List[str]]:
    """Returns a list of labels (0 for ctrl, 1 for autism) given a list of nii paths"""
    subjects = pd.read_csv(subj_data, usecols=["FILE_ID", "DX_GROUP", "SITE_ID"])
    # convert from stupid (1,2)=(AUTISM,CTRL) to (0, 1)=(CTRL, AUTISM)
    subjects["DX_GROUP"] = 2 - subjects["DX_GROUP"]
    subjects.rename(
        columns={"DX_GROUP": "label", "FILE_ID": "fid", "SITE_ID": "site"}, inplace=True
    )
    subjects.index = subjects.fid.to_list()
    subjects.drop(columns="fid")
    if "1D" in imgs[0].name:
        fids = [img.stem[: img.stem.find("_rois")] for img in imgs]
    else:
        fids = [img.stem[: img.stem.find("__")] for img in imgs]
    labels: List[int] = subjects.loc[fids].label.to_list()
    sites = subjects.loc[fids].site.to_list()
    for i, site in enumerate(sites):
        # remove some unnecessary fine-grained site distinctions like LEUVEN_1, UM_2, UM_1
        for s in ["LEUVEN", "UCLA", "UM"]:
            if s in site:
                sites[i] = s

    return labels, sites


def load_eigs(eig: Path) -> ndarray:
    return np.load(eig)  # type; ignore


def load_corrmat(path: Path) -> ndarray:
    return np.load(path)  # type; ignore


def load_corrs_from_roi_1D_file(path: Path) -> ndarray:
    return pd.read_csv(path, sep="\t").corr().to_numpy()  # type: ignore


@MEMOIZER.cache
def load_X_labels_unshuffled_from_1D() -> Tuple[ndarray, List[int], List]:
    DATA = ROOT / "data"
    SUBJ_DATA = DATA / "Phenotypic_V1_0b_preprocessed1.csv"
    CC200 = DATA / "rois_cpac_cc200"
    # CORR_FEAT_NAMES = ["r_mean", "r_sd", "r_05", "r_95", "r_max", "r_min"]
    ROI_PATHS = sorted(CC200.rglob("*.1D*"))
    labels_, sites = get_labels_sites(ROI_PATHS, SUBJ_DATA)
    feats = np.stack(
        process_map(
            load_corrs_from_roi_1D_file,
            ROI_PATHS,
            chunksize=1,
            disable=False,
            desc="Loading correlation features from 1D files",
        )
    )
    X = np.stack(feats, axis=0)
    X[np.isnan(X)] = 0  # 20-30 subjects do have NaNs, usually about 1000-3000 when occurs
    idx_h, idx_w = np.triu_indices_from(X[0, :, :], k=1)
    X = X[:, idx_h, idx_w]  # 19 900 features
    return X, labels_, sites


def load_X_labels_from_1D(
    n_features: int,
    select: Literal["mean", "sd"] = "mean",
    norm: Norm = None,
) -> Tuple[Tensor, Tensor, List, List]:
    X, labels_, sites = load_X_labels_unshuffled_from_1D()

    # shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    labels = np.array(labels_)[idx]
    sites = np.array(sites)[idx].tolist()

    if select == "mean":
        sort_idx = np.argsort(-np.abs(np.mean(X, axis=0)))
    elif select == "sd":
        sort_idx = np.argsort(-np.std(X, axis=0, ddof=1))
    else:
        raise ValueError("Invalid feature selection method")
    X = X[:, sort_idx[:n_features]]

    if norm is None:
        pass
    elif norm == "const":
        X += 1
        X /= 2  # are correlations, normalize to be positive in [0, 1]
    elif norm == "feature":
        X -= X.mean(axis=0)
        sd = X.std(axis=0, ddof=1)
        sd[np.isnan(sd)] = 1.0
        X /= sd
    elif norm == "grand":
        X -= X.mean()
        X /= X.std(ddof=1)
    else:
        raise ValueError("Invalid norm method.")

    labels = torch.tensor(labels_)
    X = torch.from_numpy(X.copy()).float()
    return X, labels, labels_, sites


@MEMOIZER.cache
def load_X_labels_unshuffled(n: int = 19900 // 2) -> Tuple[ndarray, ndarray, List]:
    DATA = ROOT / "data"
    SUBJ_DATA = DATA / "Phenotypic_V1_0b_preprocessed1.csv"
    CC200 = ROOT / "data/features_cpac/cc200"
    # CORR_FEAT_NAMES = ["r_mean", "r_sd", "r_05", "r_95", "r_max", "r_min"]
    CORR_FEAT_NAMES = ["r_mean", "r_sd"]
    CORR_FEATURE_DIRS = [CC200 / p for p in CORR_FEAT_NAMES]
    CORR_FEAT_PATHS = [sorted(p.rglob("*.npy")) for p in CORR_FEATURE_DIRS]
    labels_, sites = get_labels_sites(CORR_FEAT_PATHS[0], SUBJ_DATA)
    feats = [
        np.stack(process_map(load_corrmat, paths, chunksize=1, disable=True))
        for paths in tqdm(CORR_FEAT_PATHS, desc="Loading correlation features")
    ]
    X = np.stack(feats, axis=1)
    X[np.isnan(X)] = 0  # shouldn't be any, but just in case
    idx_h, idx_w = np.triu_indices_from(X[0, 0, :, :], k=1)
    X = X[:, :, idx_h, idx_w]  # 19 900 features
    sort_idx = np.argsort(-np.abs(np.mean(X, axis=0)), axis=1)
    largests = []
    for c in range(sort_idx.shape[0]):  # channel
        ix = sort_idx[c]
        largests.append(X[:, c, ix[:n]])
    X = np.stack(largests, axis=1)
    X += 1
    X /= 2  # are correlations, normalize to be positive in [0, 1]
    return X, labels_, sites


def load_X_labels(n: int) -> Tuple[Tensor, Tensor, List, List]:
    X, labels_, sites = load_X_labels_unshuffled(n)

    # shuffle
    idx = np.random.permutation(len(X))
    X = X[idx]
    labels = np.array(labels_)[idx]
    sites = np.array(sites)[idx].tolist()

    labels = torch.tensor(labels_)
    X = torch.from_numpy(X.copy()).float()
    return X, labels, labels_, sites
