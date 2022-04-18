import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
logging.getLogger("tensorflow").setLevel(logging.FATAL)
logging.getLogger("tensorboard").setLevel(logging.FATAL)

import sys
import traceback
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)
from uuid import UUID, uuid1
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import torch
from joblib import Memory
from numpy import ndarray
from pandas import DataFrame, Series
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    ParameterGrid,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from sklearn.svm import LinearSVC
from torch import Tensor
from torch.nn import (
    SELU,
    AvgPool1d,
    AvgPool2d,
    BatchNorm1d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    Dropout,
    Flatten,
    Identity,
    LazyLinear,
    LeakyReLU,
    Linear,
    MaxPool1d,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy, f1, precision
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal
from xgboost import DMatrix, XGBClassifier

from src.analysis.predict.hypertune import evaluate_hypertuned, hypertune_classifier

from scratch_models import GUESS, ASDDiagNet, LinearModel, Subah2021

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "lightning_logs"
MEMOIZER = Memory("__CACHE__")

Norm = Optional[Literal["const", "feature", "grand"]]


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
    return np.load(eig)


def compare_1D_eig_models() -> None:
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT / "data"
    SUBJ_DATA = DATA / "Phenotypic_V1_0b_preprocessed1.csv"
    EIG_PATHS = sorted((ROOT / "data/features_cpac/eig_full").rglob("*.npy"))
    BATCH_SIZE = 32
    WORKERS = 4
    # SLICER = slice(-150, -10)
    SLICER = slice(None)
    LEN = len(np.arange(314)[SLICER])

    labels, sites = get_labels_sites(EIG_PATHS, SUBJ_DATA)
    eigs = process_map(load_eigs, EIG_PATHS, chunksize=1, disable=True)
    eigs = torch.vstack(
        [torch.nn.ConstantPad1d((315 - len(e), 0), 0.0)(torch.from_numpy(e).T) for e in eigs]
    )
    eigs /= eigs.max(dim=1)[0].reshape(-1, 1)
    # eigs /= eigs.max()
    eigs = eigs[:, :-1]  # remove 1.0 vector
    eigs = eigs.float()[:, SLICER]
    dummy = np.empty([eigs.shape[0], 1])
    idx_train, idx_val = next(
        StratifiedShuffleSplit(n_splits=1, test_size=200).split(X=dummy, y=labels, groups=sites)
    )
    labels = torch.tensor(labels)
    x_train, x_val = eigs[idx_train], eigs[idx_val]
    y_train, y_val = labels[idx_train], labels[idx_val]
    GUESS = 0.5 + np.abs(torch.mean(y_val.float()) - 0.5)

    args = dict(batch_size=BATCH_SIZE, num_workers=WORKERS, drop_last=True)
    train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, **args)
    val_loader = DataLoader(TensorDataset(x_val, y_val), shuffle=False, **args)
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    train_args = parser.parse_known_args()[0]
    cbs = [LearningRateMonitor()]

    shared_args: Dict[str, Any] = dict(init_ch=32, depth=4, weight_decay=1e-5, max_depth=512)
    model: LightningModule
    # model = LinearModel(**shared_args)
    # model = PointModel(lr=1e-3, **shared_args)
    model = ConvModel(kernel_size=7, **shared_args)

    trainer: Trainer = Trainer.from_argparse_args(
        train_args,
        callbacks=cbs,
        max_epochs=1000,
        gpus=1,
        default_root_dir=LOGS / model.__class__.__name__,
    )
    print(model)
    trainer.fit(model, train_loader, val_loader)

    """
    >>> pd.DataFrame([len(a) for a in eigs]).describe()

    count  1035.000000
    mean    193.513043
    std      58.577864
    min      77.000000
    25%     151.000000
    50%     175.000000
    75%     235.000000
    max     315.000000
    """


def load_corrmat(path: Path) -> ndarray:
    return np.load(path)


def load_corrs_from_roi_1D_file(path: Path) -> ndarray:
    return pd.read_csv(path, sep="\t").corr().to_numpy()


@MEMOIZER.cache
def load_X_labels_unshuffled_from_1D() -> Tuple[ndarray, List[int], List]:
    ROOT = Path(__file__).resolve().parent
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

    return X, labels, labels_, sites


@MEMOIZER.cache
def load_X_labels_unshuffled(n: int = 19900 // 2) -> Tuple[ndarray, ndarray, List]:
    ROOT = Path(__file__).resolve().parent
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


def test_kfold() -> None:
    global GUESS
    X, labels, labels_, sites = load_X_labels()
    dummy = np.empty([len(X), 1])
    kf = StratifiedKFold(n_splits=5).split(X=dummy, y=labels_, groups=sites)
    # all_idx_train, idx_test = next(
    #     StratifiedShuffleSplit(n_splits=1, test_size=64).split(X=dummy, y=labels_, groups=sites)
    # )
    all_results = []
    accs, acc_deltas, f1s, losses = [], [], [], []
    for i, (all_idx_train, idx_test) in enumerate(kf):
        print(f"Fold {i}:")

        # StratifiedKFold()
        X_train_all, y_train_all = X[all_idx_train], labels[all_idx_train]
        train_sites = np.array(sites)[all_idx_train]

        idx_train, idx_val = next(
            StratifiedShuffleSplit(n_splits=1, test_size=64).split(
                X=np.arange(len(X_train_all)), y=y_train_all, groups=train_sites
            )
        )

        x_train, x_val = X_train_all[idx_train], X_train_all[idx_val]
        y_train, y_val = y_train_all[idx_train], y_train_all[idx_val]
        x_test, y_test = X[idx_test], labels[idx_test]
        GUESS = 0.5 + np.abs(torch.mean(y_val.float()) - 0.5)

        args = dict(batch_size=BATCH_SIZE, num_workers=WORKERS, drop_last=True)
        train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, **args)
        val_loader = DataLoader(TensorDataset(x_val, y_val), shuffle=False, **args)
        test_loader = DataLoader(TensorDataset(x_test, y_test), shuffle=False, **args)
        parser = ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        train_args = parser.parse_known_args()[0]
        cbs = [LearningRateMonitor(), ModelCheckpoint(monitor="val/acc", mode="max")]

        shared_args: Dict[str, Any] = dict(
            init_ch=32, depth=4, weight_decay=0, max_channels=256, lr=LR
        )
        model: LightningModule
        # model = LinearModel(**shared_args)
        # model = PointModel(lr=1e-3, **shared_args)
        model = CorrNet(**shared_args)

        trainer: Trainer = Trainer.from_argparse_args(
            train_args,
            callbacks=cbs,
            enable_model_summary=False,
            log_every_n_steps=23,
            max_steps=2000,
            gpus=1,
            default_root_dir=LOGS / f"corrs_test_logs/{model.__class__.__name__}/kfold",
        )
        trainer.fit(model, train_loader, val_loader)
        results = trainer.test(model, test_loader, ckpt_path="best")
        accs.append(results[0]["test/acc"])
        acc_deltas.append(results[0]["test/acc+"])
        f1s.append(results[0]["test/f1"])
        losses.append(results[0]["test/loss"])
        all_results.append(results)
        torch.cuda.empty_cache()
        # break
    print("\n\nFinal results:")
    df = pd.DataFrame({"acc": accs, "acc+": acc_deltas, "f1": f1s, "loss": losses})
    print(df.describe().T.drop(columns="count").to_markdown(tablefmt="simple", floatfmt="0.3f"))


def test_split(
    n_features: int = 19900, feat_select: Literal["mean", "sd"] = "mean", norm: Norm = None
) -> None:
    X, labels, labels_, sites = load_X_labels_from_1D(
        n_features=n_features,
        select=feat_select,
        norm=norm,
    )
    dummy = np.empty([len(X), 1])
    idx_train, idx_val = next(
        StratifiedShuffleSplit(n_splits=1, test_size=256).split(X=dummy, y=labels_, groups=sites)
    )
    x_train, x_val = X[idx_train], X[idx_val]
    y_train, y_val = labels[idx_train], labels[idx_val]
    sites_val = np.array(sites)[idx_val]

    g = torch.abs(torch.mean(y_val.float()))
    guess = torch.max(g, 1 - g)

    args = dict(batch_size=BATCH_SIZE, num_workers=WORKERS, drop_last=True)
    train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, **args)
    val_loader = DataLoader(TensorDataset(x_val, y_val), shuffle=False, **args)
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    train_args = parser.parse_known_args()[0]
    cbs = [
        LearningRateMonitor(),
        ModelCheckpoint(monitor="val/acc", mode="max"),
        EarlyStopping(monitor="val/acc+", patience=30, mode="max", divergence_threshold=-4.0),
    ]

    uuid = uuid1().hex
    model: LightningModule
    model = LinearModel(uuid=uuid, **SHARED_ARGS)
    # model = PointModel(lr=1e-3, **shared_args)
    # model = CorrNet(**SHARED_ARGS)
    # model = SharedAutoEncoder(**SHARED_ARGS)
    # model = ASDDiagNet(**SHARED_ARGS, guess=guess)
    # model = Subah2021(**SHARED_ARGS, guess=guess)
    wd = SHARED_ARGS["weight_decay"]
    root_dir = (
        LOGS
        / f"corrs_test_logs/{model.__class__.__name__}/holdout/n={n_features}/sel={feat_select}/norm={norm}/wd={wd}/{uuid}"
    )

    trainer: Trainer = Trainer.from_argparse_args(
        train_args,
        callbacks=cbs,
        enable_model_summary=False,
        log_every_n_steps=64,
        max_epochs=MAX_EPOCHS,
        max_steps=MAX_STEPS,
        gpus=1,
        default_root_dir=root_dir,
    )
    # result = trainer.tuner.lr_find(model, train_loader, val_loader, num_training=200)
    # result.plot(suggest=True, show=True)
    filterwarnings("ignore", message="The dataloader, val_dataloader 0, does not have many workers")
    filterwarnings("ignore", message="The dataloader, train_dataloader, does not have many workers")
    filterwarnings("ignore", message="The number of training samples")
    print("=" * 150)
    sites, counts = np.unique(sites_val, return_counts=True)
    print(f"Val site distribution: ({len(sites)} sites)\n")
    df = DataFrame(data=counts, columns=["N"], index=sites).T
    print(df.to_markdown(tablefmt="simple"))
    print(f"\nGuess acc: {guess}")
    print("=" * 150)

    trainer.fit(model, train_loader, val_loader)


def test_xgboost() -> None:
    X, labels, labels_, sites = load_X_labels()
    dummy = np.empty([len(X), 1])
    idx_train, idx_val = next(
        StratifiedShuffleSplit(n_splits=1, test_size=256).split(X=dummy, y=labels_, groups=sites)
    )
    x_train, x_val = X[idx_train], X[idx_val]
    y_train, y_val = labels[idx_train], labels[idx_val]
    x_train = x_train.reshape(x_train.shape[0], -1)
    x_val = x_val.reshape(x_val.shape[0], -1)

    GUESS = 0.5 + np.abs(torch.mean(y_val.float()) - 0.5)
    model = XGBClassifier(n_jobs=-1, objective="binary:logistic", use_label_encoder=False)
    result = hypertune_classifier("xgb", x_train, y_train, n_trials=100, cv_method=3, verbosity=10)
    print(result)
    print("Best params:")
    print(result.best_params)
    print("Tuning acc: ")
    print(f"{result.val_acc} ({result.val_acc - GUESS})")

    res = evaluate_hypertuned(
        result, cv_method=None, X_train=x_train, y_train=y_train, X_test=x_val, y_test=y_val
    )
    print(res)


@dataclass
class LrArgs:
    X: ndarray
    labels: ndarray
    sites: List[str]
    args: Dict


def eval_lr(lr_args: LrArgs) -> float:
    X, labels, sites = lr_args.X, lr_args.labels, lr_args.sites
    args = lr_args.args
    lr = LogisticRegression(verbose=0, **args)
    score = np.mean(cross_val_score(lr, X, labels, groups=sites, cv=5, n_jobs=5))
    print(args, score)
    return score, args


def test_log_reg() -> None:
    X, labels, labels_, sites = load_X_labels()
    X = X.reshape(X.shape[0], -1)
    # N = 10
    # X = X[:N, :N]
    # labels = labels[:N]
    # sites = sites[:N]
    grid = list(
        ParameterGrid(
            dict(
                penalty=["l2"],
                solver=["liblinear"],
                dual=[True],
                C=[1e5, 1e4, 1e3, 1e2, 10, 1, 0.5, 0.1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6],
                max_iter=[400],
            )
        )
    )
    kf = StratifiedKFold()
    args = [LrArgs(X, labels.numpy(), sites, arg) for arg in grid]
    scores, args = list(zip(*process_map(eval_lr, args, max_workers=4)))
    print(
        DataFrame(scores)
        .describe()
        .T.drop(columns="count")
        .rename(index=lambda x: "acc")
        .to_markdown(tablefmt="simple")
    )


if __name__ == "__main__":
    N = int((200 ** 2 - 200) / 2)  # upper triangle of 200x200 matrix where diagonals are 1
    # n = N // 8
    n = N
    # n = 2000
    # BATCH_SIZE = 32
    BATCH_SIZE = 32
    # LR = 3e-4
    # LR = 1e-2
    LR = 3e-4
    # LR = 1e-3
    WORKERS = 0
    # MAX_STEPS = 20000
    MAX_STEPS = 3000
    MAX_EPOCHS = 150

    # LinearModel
    SHARED_ARGS: Dict[str, Any] = dict(
        in_features=n, init_ch=16, depth=2, max_channels=512, dropout=0.0, weight_decay=0, lr=LR
    )

    # Eslami ASDDiagNet
    # SHARED_ARGS: Dict[str, Any] = dict(
    #     in_features=n,
    #     bottleneck=250,
    #     weight_decay=0,
    #     lr=LR,
    # )

    # Subah 2021
    # SHARED_ARGS: Dict[str, Any] = dict(
    #     in_features=n,
    #     lr=LR,
    #     weight_decay=1e-6,
    #     dropout=0.8,
    # )
    # FEAT_SELECT = "mean"
    FEAT_SELECT = "sd"
    NORM: Norm = "feature"  # this is the only one that works
    # NORM: Norm = "const"
    # NORM: Norm = "grand"
    # test_kfold()
    # test_split(n_features=n, feat_select=FEAT_SELECT, norm=NORM)

    # test all linear models
    # grid = list(
    #     ParameterGrid(
    #         dict(
    #             in_features=[n // 2, n // 4, 2000],
    #             init_ch=[16, 64],
    #             depth=[2, 4, 8],
    #             max_channels=[64, 256],
    #             dropout=[0.0, 0.25, 0.5, 0.75],
    #             weight_decay=[0.0, 1e-4],
    #         )
    #     )
    # )

    # test winning linear models with repeats
    grid = [
        dict(
            in_features=9950,
            init_ch=16,
            depth=8,
            max_channels=256,
            dropout=0.75,
            weight_decay=0.0,
        ),
        dict(
            in_features=9950,
            init_ch=16,
            depth=8,
            max_channels=256,
            dropout=0.5,
            weight_decay=1e-4,
        ),
        dict(
            in_features=2000,
            init_ch=64,
            depth=2,
            max_channels=256,
            dropout=0.75,
            weight_decay=1e-4,
        )
    ]

    for i, params in enumerate(grid):
        print(f"Iteration {i} of {len(grid)}.")
        print(f"Testing params: {params}")
        SHARED_ARGS.update(params)
        # SHARED_ARGS["weight_decay"] = params["weight_decay"]
        # SHARED_ARGS["in_features"] = params["weight_decay"]
        n = params["in_features"]
        for _ in range(5):
            try:
                test_split(n_features=n, feat_select=FEAT_SELECT, norm=NORM)
            except Exception as e:
                traceback.print_exc()
