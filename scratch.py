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
from numpy import ndarray
from pandas import DataFrame, Series
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import (
    Callback,
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from sklearn.model_selection import StratifiedShuffleSplit
from torch import Tensor
from torch.nn import (
    AvgPool1d,
    AvgPool2d,
    BatchNorm1d,
    Conv1d,
    Flatten,
    LeakyReLU,
    Linear,
    MaxPool1d,
    Module,
    Sequential,
)
from torch.nn.functional import binary_cross_entropy_with_logits
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.functional import accuracy, f1, precision
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "lightning_logs"


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


class GlobalAveragePool1D(Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: Tensor) -> Tensor:
        # x.shape == (B, C, len)
        return torch.mean(x, dim=-1)


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

    def configure_optimizers(self) -> Any:
        opt = Adam(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        step = 500
        lr_decay = 0.9
        sched = StepLR(opt, step_size=step, gamma=lr_decay)
        return dict(
            optimizer=opt,
            lr_scheduler=dict(
                scheduler=sched,
                interval="step",
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
        layers: List[Module] = [Conv(in_channels=1, out_channels=init_ch, kernel_size=kernel_size)]
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


if __name__ == "__main__":
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
