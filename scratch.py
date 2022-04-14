import sys
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
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    StratifiedKFold,
    StratifiedShuffleSplit,
)
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
    LazyLinear,
    LeakyReLU,
    Linear,
    MaxPool1d,
    MaxPool2d,
    Module,
    ReLU,
    Sequential,
)
from torch.nn.functional import binary_cross_entropy_with_logits
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

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "lightning_logs"
MEMOIZER = Memory("__CACHE__")


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


class Lin(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model = Sequential(
            Linear(in_channels, out_channels, bias=True),
            LeakyReLU(),
            BatchNorm1d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x = self.model(x)
        return x


class PointLinear(Module):
    def __init__(self, out_channels: int) -> None:
        super().__init__()
        self.model = Sequential(
            Linear(4, out_channels, bias=True),
            LeakyReLU(),
            BatchNorm1d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        x_mean = torch.mean(x, dim=1)
        x_min = torch.min(x, dim=1)[0]
        x_max = torch.max(x, dim=1)[0]
        x_sd = torch.std(x, dim=1)
        x = torch.stack([x_mean, x_min, x_max, x_sd], dim=1)
        x = self.model(x)
        return x


class Conv(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.model = Sequential(
            Conv1d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                # padding="same",
                padding=0,
                bias=True,
            ),
            LeakyReLU(),
            BatchNorm1d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # needs x.shape == (B, C, seq_length)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.model(x)
        return x


class Conv2(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.model = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                # padding="same",
                padding=0,
                bias=True,
            ),
            LeakyReLU(),
            BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # needs x.shape == (B, C, seq_length)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.model(x)
        return x


class GlobalAveragePool1D(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, C, len)
        return torch.mean(x, dim=-1)


class GlobalAveragePool2D(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, C, H, W)
        return torch.mean(x, dim=(2, 3))


class TrainingMixin(LightningModule, ABC):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()

    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs) -> Any:
        return self.shared_step(batch, "train")

    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ) -> Optional[Any]:
        self.shared_step(batch, "val")

    def test_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ) -> Optional[Any]:
        self.shared_step(batch, "test")

    def configure_optimizers(self) -> Any:
        opt = Adam(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        # step = 500
        step = 1
        lr_decay = 0.95
        sched = StepLR(opt, step_size=step, gamma=lr_decay)
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=sched,
                # interval="step",
                interval="epoch",
            ),
        )

    def shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Tensor:
        x, target = batch
        preds = self.model(x)
        loss = binary_cross_entropy_with_logits(preds.squeeze(), target.float())
        acc = accuracy(preds, target) - GUESS
        f1_score = f1(preds, target)
        self.log(f"{phase}/loss", loss)
        self.log(f"{phase}/acc+", acc, prog_bar=True)
        # self.log(f"{phase}/prec", prec, prog_bar=True)
        self.log(f"{phase}/f1", f1_score, prog_bar=True)
        if phase == "val":
            self.log(f"{phase}/acc", acc + GUESS, prog_bar=True)
        return loss


class LinearModel(TrainingMixin):
    def __init__(
        self,
        init_ch: int = 16,
        depth: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_depth: int = 512,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        layers: List[Module] = [Lin(LEN, init_ch)]
        ch = init_ch
        out = ch
        for _ in range(depth - 1):
            out = min(max_depth, out * 2)
            layers.append(Lin(ch, out))
            ch = out
        layers.append(Linear(out, 1, bias=True))
        self.model = Sequential(*layers)


class PointModel(TrainingMixin):
    def __init__(
        self,
        init_ch: int = 16,
        depth: int = 4,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_depth: int = 512,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        layers: List[Module] = [PointLinear(init_ch)]
        ch = init_ch
        out = ch
        for _ in range(depth - 1):
            out = min(max_depth, out * 2)
            layers.append(Lin(ch, out))
            ch = out
        layers.append(Linear(out, 1, bias=True))
        self.model = Sequential(*layers)


class ConvModel(TrainingMixin):
    def __init__(
        self,
        in_channels: int = 1,
        init_ch: int = 16,
        depth: int = 4,
        kernel_size: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_depth: int = 512,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        layers: List[Module] = [
            Conv(in_channels=in_channels, out_channels=init_ch, kernel_size=kernel_size)
        ]
        ch = init_ch
        out = ch
        for _ in range(depth - 1):
            out = min(max_depth, out * 2)
            layers.append(Conv(ch, out, kernel_size=kernel_size))
            ch = out
        # will have shape (B, out, LEN)
        layers.append(GlobalAveragePool1D())
        layers.append(Linear(out, 1, bias=True))
        self.model = Sequential(*layers)


class Conv2dModel(TrainingMixin):
    def __init__(
        self,
        in_channels: int = 1,
        init_ch: int = 16,
        depth: int = 4,
        kernel_size: int = 3,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        max_depth: int = 512,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        layers: List[Module] = [
            Conv2(in_channels=in_channels, out_channels=init_ch, kernel_size=kernel_size)
        ]
        ch = init_ch
        out = ch
        for _ in range(depth - 1):
            out = min(max_depth, out * 2)
            layers.append(Conv2(ch, out, kernel_size=kernel_size))
            ch = out
        # will have shape (B, out, LEN)
        layers.append(GlobalAveragePool2D())
        layers.append(Linear(out, 1, bias=True))
        self.model = Sequential(*layers)


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


class CorrCell(Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.model = Sequential(
            Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=1,
                # padding="same",
                padding=0,
                bias=True,
            ),
            LeakyReLU(),
            BatchNorm2d(out_channels),
        )

    def forward(self, x: Tensor) -> Tensor:
        # needs x.shape == (B, C, seq_length)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.model(x)
        return x


class CorrPool(Module):
    def __init__(self, in_channels: int, spatial_in: Tuple[int, int]) -> None:
        super().__init__()
        self.conv = Conv2d(
            in_chanels=in_channels, out_channels=in_channels // 2, kernel_size=spatial_in, padding=0
        )

    def forward(self, x: Tensor) -> Tensor:
        # needs x.shape == (B, C, seq_length)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x1 = self.maxpool(x)
        x2 = self.avgpool(x)
        x = torch.cat([x1, x2])
        return x


class Thresholder(Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.relu = ReLU()
        self.attention = Parameter(torch.randn([1, in_channels, 200, 200]), requires_grad=True)
        self.thresh = Parameter(torch.randn([1, in_channels, 1, 1]), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # needs x.shape == (B, C, seq_length)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = x * 2 - 1  # send back to [-1, 1]
        mask = torch.abs(x) - torch.abs(
            self.thresh
        )  # makes correlations > self.thresh all that matter
        mask = self.relu(mask)
        x = x * mask
        x = torch.mul(torch.sigmoid(self.attention), x)
        return x


class CorrInput(Module):
    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.relu = ReLU()
        self.attention = Parameter(torch.randn([1, in_channels, 200, 200]), requires_grad=True)
        self.thresh = Parameter(torch.randn([1, in_channels, 1, 1]), requires_grad=True)

    def forward(self, x: Tensor) -> Tensor:
        # needs x.shape == (B, C, seq_length)
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = x * 2 - 1  # send back to [-1, 1]
        mask = torch.abs(x) - torch.abs(
            self.thresh
        )  # makes correlations > self.thresh all that matter
        mask = self.relu(mask)
        x = x * mask
        x = torch.mul(torch.sigmoid(self.attention), x)
        return x


class CorrNet(TrainingMixin):
    def __init__(
        self,
        init_ch: int = 16,
        depth: int = 4,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        max_channels: int = 512,
        dropout: float = 0.6,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        # layers: List[Module] = [
        #     Conv2(in_channels=2, out_channels=init_ch, kernel_size=1)
        # ]
        layers: List[Module] = [
            # Thresholder(in_channels=6),
            Conv(in_channels=6, out_channels=init_ch, kernel_size=N),
            Flatten(),
            # LazyLinear(out_features=init_ch, bias=False),
            # BatchNorm1d(init_ch),
            # LeakyReLU(inplace=True),
        ]
        ch = init_ch
        out = ch
        for _ in range(depth - 1):
            out = min(max_channels, out * 2)
            # layers.append(CorrCell(ch, out))
            layers.append(Linear(ch, out))
            layers.append(BatchNorm1d(out))
            layers.append(LeakyReLU(inplace=True))
            layers.append(Dropout(p=dropout))
            ch = out
        # will have shape (B, out, LEN)
        # layers.append(GlobalAveragePool2D())
        layers.append(Linear(out, 1, bias=True))
        self.model = Sequential(*layers)

    def shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Tensor:
        x, target = batch
        preds = self.model(x)
        loss = binary_cross_entropy_with_logits(preds.squeeze(), target.float())

        self.log(f"{phase}/loss", loss)
        # self.log(f"{phase}/prec", prec, prog_bar=True)
        if phase in ["val", "test"]:
            acc = accuracy(preds, target) - GUESS
            f1_score = f1(preds, target)
            self.log(f"{phase}/acc+", acc, prog_bar=True)
            self.log(f"{phase}/acc", acc + GUESS, prog_bar=True)
            self.log(f"{phase}/f1", f1_score, prog_bar=True)
        return loss


@MEMOIZER.cache
def load_X_labels_unshuffled() -> Tuple[ndarray, ndarray, List]:
    ROOT = Path(__file__).resolve().parent
    DATA = ROOT / "data"
    SUBJ_DATA = DATA / "Phenotypic_V1_0b_preprocessed1.csv"
    CC200 = ROOT / "data/features_cpac/cc200"
    CORR_FEAT_NAMES = ["r_mean", "r_sd", "r_05", "r_95", "r_max", "r_min"]
    CORR_FEATURE_DIRS = [CC200 / p for p in CORR_FEAT_NAMES]
    CORR_FEAT_PATHS = [sorted(p.rglob("*.npy")) for p in CORR_FEATURE_DIRS]
    labels_, sites = get_labels_sites(CORR_FEAT_PATHS[0], SUBJ_DATA)
    feats = [
        np.stack(process_map(load_corrmat, paths, chunksize=1, disable=True))
        for paths in tqdm(CORR_FEAT_PATHS, desc="Loading correlation features")
    ]
    X = np.stack(feats, axis=1)
    X[np.isnan(X)] = 0  # shouldn't be any, but just in case
    X += 1
    X /= 2  # are correlations, normalize to be positive in [0, 1]
    idx_h, idx_w = np.triu_indices_from(X[0, 0, :, :], k=1)
    X = X[:, :, idx_h, idx_w]  # 19 900 features
    return X, labels_, sites


def load_X_labels() -> Tuple[Tensor, Tensor, List, List]:
    X, labels_, sites = load_X_labels_unshuffled()

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


def test_split() -> None:
    global GUESS
    X, labels, labels_, sites = load_X_labels()
    dummy = np.empty([len(X), 1])
    idx_train, idx_val = next(
        StratifiedShuffleSplit(n_splits=1, test_size=256).split(X=dummy, y=labels_, groups=sites)
    )
    x_train, x_val = X[idx_train], X[idx_val]
    y_train, y_val = labels[idx_train], labels[idx_val]
    GUESS = 0.5 + np.abs(torch.mean(y_val.float()) - 0.5)

    args = dict(batch_size=BATCH_SIZE, num_workers=WORKERS, drop_last=True)
    train_loader = DataLoader(TensorDataset(x_train, y_train), shuffle=True, **args)
    val_loader = DataLoader(TensorDataset(x_val, y_val), shuffle=False, **args)
    parser = ArgumentParser()
    parser = Trainer.add_argparse_args(parser)
    train_args = parser.parse_known_args()[0]
    cbs = [LearningRateMonitor(), ModelCheckpoint(monitor="val/acc", mode="max")]

    model: LightningModule
    # model = LinearModel(**shared_args)
    # model = PointModel(lr=1e-3, **shared_args)
    model = CorrNet(**SHARED_ARGS)

    trainer: Trainer = Trainer.from_argparse_args(
        train_args,
        callbacks=cbs,
        enable_model_summary=False,
        log_every_n_steps=64,
        max_steps=6000,
        gpus=1,
        default_root_dir=LOGS / f"corrs_test_logs/{model.__class__.__name__}/holdout",
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    N = int((200 ** 2 - 200) / 2)  # upper triangle of 200x200 matrix where diagonals are 1
    GUESS = None
    BATCH_SIZE = 32
    LR = 3e-4
    # LR = 1e-3
    WORKERS = 4
    SHARED_ARGS: Dict[str, Any] = dict(
        init_ch=256, depth=8, weight_decay=0, max_channels=512, lr=LR, dropout=0.4
    )
    X, labels, labels_, sites = load_X_labels()
    dummy = np.empty([len(X), 1])
    idx_train, idx_val = next(
        StratifiedShuffleSplit(n_splits=1, test_size=256).split(X=dummy, y=labels_, groups=sites)
    )
    x_train, x_val = X[idx_train], X[idx_val]
    y_train, y_val = labels[idx_train], labels[idx_val]

    GUESS = 0.5 + np.abs(torch.mean(y_val.float()) - 0.5)
    model = XGBClassifier(n_jobs=-1, objective="binary:logistic", use_label_encoder=False)
    train = DMatrix(x_train.reshape(x_train.shape[0], -1), y_train)
    clf = GridSearchCV(model, {'max_depth': [2, 4, 6], 'n_estimators': [100, 200]}, verbose=1, n_jobs=1, cv=3)
    clf.fit(x_train.reshape(x_train.shape[0], -1), y_train, eval_metric="logloss")
    clf.score(x_val.reshape(x_val.shape[0], -1), y_val)
    # test_kfold()
    # test_split()
