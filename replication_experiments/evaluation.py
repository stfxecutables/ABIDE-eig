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

import traceback
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from uuid import uuid1
from warnings import filterwarnings

import numpy as np
import pandas as pd
import torch
from numpy import ndarray
from pandas import DataFrame
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    ParameterGrid,
    StratifiedKFold,
    StratifiedShuffleSplit,
    cross_val_score,
)
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

from replication_experiments.loading import (
    get_labels_sites,
    load_eigs,
    load_X_labels,
    load_X_labels_from_1D,
)
from replication_experiments.models import LinearModel
from src.analysis.predict.hypertune import evaluate_hypertuned, hypertune_classifier

LOGS = ROOT / "replication_experiments/logs"

Norm = Optional[Literal["const", "feature", "grand"]]


def compare_1D_eig_models() -> None:
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT / "data"
    SUBJ_DATA = DATA / "Phenotypic_V1_0b_preprocessed1.csv"
    EIG_PATHS = sorted((ROOT / "data/features_cpac/eig_full").rglob("*.npy"))
    BATCH_SIZE = 32
    WORKERS = 4
    # SLICER = slice(-150, -10)
    SLICER = slice(None)
    len(np.arange(314)[SLICER])

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
    0.5 + np.abs(torch.mean(y_val.float()) - 0.5)

    args: Dict = dict(batch_size=BATCH_SIZE, num_workers=WORKERS, drop_last=True)
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


def test_kfold() -> None:
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
        guess = 0.5 + np.abs(torch.mean(y_val.float()) - 0.5)

        args: Dict = dict(batch_size=BATCH_SIZE, num_workers=WORKERS, drop_last=True)
        train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, **args)
        val_loader = DataLoader(TensorDataset(x_val, y_val), shuffle=False, **args)
        test_loader = DataLoader(TensorDataset(x_test, y_test), shuffle=False, **args)
        parser = ArgumentParser()
        parser = Trainer.add_argparse_args(parser)
        train_args = parser.parse_known_args()[0]
        cbs = [LearningRateMonitor(), ModelCheckpoint(monitor="val/acc", mode="max")]

        shared_args: Dict[str, Any] = dict(
            init_ch=32, depth=4, max_channels=256, weight_decay=0, lr=LR, guess=guess
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
    n_features: int = 19900,
    feat_select: Literal["mean", "sd"] = "mean",
    norm: Norm = None,
    batch_size: int = 16,
    model_args: Dict = None,
    trainer_args: Dict = None,
    logdirname: str = None,
) -> None:
    if model_args is None:
        model_args = {}
    if trainer_args is None:
        raise ValueError("Must specify at least MAX_STEPS or MAX_EPOCHS")
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

    guess_ = torch.abs(torch.mean(y_val.float()))  # two classes, so mean is proportion of class=1
    guess = torch.max(guess_, 1 - guess_)  # accuracy if you just guess the largest class

    args: Dict = dict(batch_size=batch_size, num_workers=0, drop_last=True)
    train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, **args)
    val_loader = DataLoader(TensorDataset(x_val, y_val), shuffle=False, **args)
    cbs = [
        LearningRateMonitor(),
        ModelCheckpoint(monitor="val/acc", mode="max", filename=r"{epoch}-{val/acc:.2f}"),
        EarlyStopping(monitor="val/acc+", patience=30, mode="max", divergence_threshold=-4.0),
    ]

    uuid = uuid1().hex
    model: LightningModule
    model = LinearModel(uuid=uuid, **model_args)
    # model = PointModel(lr=1e-3, **shared_args)
    # model = CorrNet(**SHARED_ARGS)
    # model = SharedAutoEncoder(**SHARED_ARGS)
    # model = ASDDiagNet(**SHARED_ARGS, guess=guess)
    # model = Subah2021(**SHARED_ARGS, guess=guess)
    wd = model_args["weight_decay"]
    dirname = logdirname or model.__class__.__name__
    root_dir = (
        LOGS / f"{dirname}/holdout/n={n_features}/sel={feat_select}/norm={norm}/wd={wd}/{uuid}"
    )

    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    train_args = parser.parse_known_args()[0]
    trainer: Trainer = Trainer.from_argparse_args(
        train_args,
        callbacks=cbs,
        enable_model_summary=False,
        log_every_n_steps=64,
        gpus=1,
        default_root_dir=root_dir,
        **trainer_args,
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

    result = hypertune_classifier("xgb", x_train, y_train, n_trials=100, cv_method=3, verbosity=10)
    print(result)
    print("Best params:")
    print(result.best_params)
    print("Tuning acc: ")
    guess = 0.5 + np.abs(torch.mean(y_val.float()) - 0.5)
    print(f"{result.val_acc} ({result.val_acc - guess})")

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


def eval_lr(lr_args: LrArgs) -> Tuple[float, Dict]:
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
    StratifiedKFold()
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

    # test winning linear models with repeats (only small n_features shows consistent behaviour)
    # grid = [
    #     dict(
    #         in_features=9950,
    #         init_ch=16,
    #         depth=8,
    #         max_channels=256,
    #         dropout=0.75,
    #         weight_decay=0.0,
    #     ),
    #     dict(
    #         in_features=9950,
    #         init_ch=16,
    #         depth=8,
    #         max_channels=256,
    #         dropout=0.5,
    #         weight_decay=1e-4,
    #     ),
    #     dict(
    #         in_features=2000,
    #         init_ch=64,
    #         depth=2,
    #         max_channels=256,
    #         dropout=0.75,
    #         weight_decay=1e-4,
    #     )
    # ]

    for i, model_args in enumerate(grid):
        print(f"Iteration {i} of {len(grid)}.")
        print(f"Testing model with args: {model_args}")
        SHARED_ARGS.update(model_args)
        # SHARED_ARGS["weight_decay"] = model_args["weight_decay"]
        # SHARED_ARGS["in_features"] = model_args["weight_decay"]
        n = model_args["in_features"]
        for _ in range(5):
            try:
                test_split(n_features=n, feat_select=FEAT_SELECT, norm=NORM, model_args=model_args)
            except Exception:
                traceback.print_exc()
