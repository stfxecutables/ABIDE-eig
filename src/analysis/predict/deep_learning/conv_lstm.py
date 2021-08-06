# fmt: off
from logging import warn
from pathlib import Path

from torch import random

import sys  # isort:skip


from torch.utils.data.dataset import TensorDataset  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor
from torch.nn import BatchNorm3d, BCEWithLogitsLoss, Conv3d, Linear, LSTMCell, Module, PReLU, ReLU
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.functional import accuracy
from typing_extensions import Literal

from src.analysis.predict.deep_learning.dataloader import FmriDataset
from src.analysis.predict.deep_learning.layers import GlobalAveragePooling

INPUT_SHAPE = (175, 61, 73, 61)
BATCH_SIZE = 10

"""
Notes from https://ieeexplore.ieee.org/document/8363798

For our purpose, a modified C3D model (number of kernels were changed and fewer layers were used)
was trained to classify the preprocessed fMRI sliding window 2 channel images. The network
architecture is shown in Fig. 2. It has 6 convolutional, 4 max-pooling and 2 fully connected layers,
followed by a sigmoid output layer. The number of kernels and the layer types are denoted in each
box. All 3D convolutional kernels were 3 × 3 × 3 with stride 1 in all dimensions. All the pooling
kernels were 2 × 2 × 2. Binary cross entropy was used as the loss function. Dropout layers were
added with ratio 0.5 after the first and the sec- ond max pooling layers and ratio 0.65 after the
third and the fourth max pooling layers. L2 regularization with regulari- sation 0.01 was used in
each fully connected layer to avoid overfitting
"""


class Downsample(Module):
    def __init__(self):
        super().__init__()


class ResBlock(Module):
    def __init__(self):
        super().__init__()
        ch = INPUT_SHAPE[0]
        KERNEL = 3
        STRIDE = 2
        DILATION = 3
        PADDING = 3
        conv_args: Dict = dict(
            kernel_size=KERNEL,
            stride=STRIDE,
            dilation=DILATION,
            padding=PADDING,
            bias=False,
            groups=1,
        )
        # BNorm  after PReLU, see https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
        self.conv1 = Conv3d(in_channels=ch, out_channels=ch, **conv_args)
        self.relu1 = PReLU()
        self.norm1 = BatchNorm3d(ch)
        self.conv2 = Conv3d(in_channels=ch, out_channels=ch, **conv_args)
        self.relu2 = PReLU()
        self.norm2 = BatchNorm3d(ch)


class ConvToLSTM(Module):
    """Use strided convolutions to learn optimal downsampling to a sequence,
    then feed that sequence to an LSTM.

    Parameters
    ----------
    param1: type1
        desc1

    Returns
    -------
    val1: Any

    Notes
    -----
    We start with an fMRI image of shape T, H, W, D (175, 61, 73, 61). Each conv
    reduces the shape so we get:
    input: (175, 61, 73, 61)
    conv1: (175, 30, 36, 30)
    conv2: (175, 7, 9, 7)
    conv3: (175, 3, 4, 3)
    conv4: (175, 1, 2, 1)
    """

    def __init__(self):
        super().__init__()
        ch = INPUT_SHAPE[0]
        conv_args: Dict = dict(in_channels=ch, out_channels=ch, kernel_size=2, stride=2, bias=False)
        self.conv1 = Conv3d(**conv_args)
        self.conv2 = Conv3d(**conv_args)


# based on
# https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py
# https://github.com/SreenivasVRao/ConvGRU-ConvLSTM-PyTorch/blob/master/convlstm.py
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py


class MemTestCNN(LightningModule):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # self.save_hyperparameters()
        ch = INPUT_SHAPE[0]
        KERNEL = 3
        STRIDE = 2
        DILATION = 3
        PADDING = 3
        conv_args: Dict = dict(
            kernel_size=KERNEL,
            stride=STRIDE,
            dilation=DILATION,
            padding=PADDING,
            bias=False,
            groups=ch,
        )
        # BNorm  after PReLU, see https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
        self.conv1 = Conv3d(in_channels=ch, out_channels=ch, **conv_args)
        self.relu1 = PReLU()
        self.norm1 = BatchNorm3d(ch)
        self.conv2 = Conv3d(in_channels=ch, out_channels=ch, **conv_args)
        self.relu2 = PReLU()
        self.norm2 = BatchNorm3d(ch)
        self.conv3 = Conv3d(in_channels=ch, out_channels=ch, **conv_args)
        self.relu3 = PReLU()
        self.norm3 = BatchNorm3d(ch)
        self.conv4 = Conv3d(in_channels=ch, out_channels=ch, **conv_args)
        self.relu4 = PReLU()
        self.norm4 = BatchNorm3d(ch)
        self.conv5 = Conv3d(in_channels=ch, out_channels=ch, **conv_args)
        self.relu5 = PReLU()
        self.norm5 = BatchNorm3d(ch)
        self.gap = GlobalAveragePooling()
        self.linear = Linear(in_features=1400, out_features=1, bias=True)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.norm3(x)
        x = self.conv4(x)
        x = self.relu4(x)
        x = self.norm4(x)
        x = self.conv5(x)
        x = self.relu5(x)
        x = self.norm5(x)
        # print("Shape after convs: ", x.shape)
        # x = self.gap(x)
        x = x.reshape([x.size(0), -1])  # flatten
        # print("Reshaped: ", x.shape)
        # sys.exit()
        x = self.linear(x)
        return x

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        acc, loss = self.inner_step(batch)
        self.log("train_acc", acc, prog_bar=True, on_step=True, on_epoch=True)
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    @no_type_check
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        acc, loss = self.inner_step(batch)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    @no_type_check
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        acc, loss = self.inner_step(batch)
        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self) -> Any:
        return torch.optim.Adam(self.parameters())

    def inner_step(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        loss: Tensor
        x, y_true = batch
        y_pred = self(x)
        criterion = BCEWithLogitsLoss()
        loss = criterion(y_pred, y_true)
        acc = accuracy(torch.sigmoid(y_pred.squeeze()), y_true.int())
        return acc, loss


def random_data() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # if our model is flexible enough we should be  able to overfit random data
    # we can!
    print("Generating class 1 training data")
    x1_train = torch.rand([20, *INPUT_SHAPE])
    print("Generating class 2 training data")
    x2_train = torch.rand([20, *INPUT_SHAPE])
    x_train = torch.cat([x1_train, x2_train])
    print("Normalizing")
    x_train -= torch.mean(x_train)
    y_train = torch.cat([torch.zeros(20), torch.ones(20)])
    print("Generating class 1 validation data")
    x1_val = torch.rand([5, *INPUT_SHAPE])
    print("Generating class 2 validation data")
    x2_val = torch.rand([5, *INPUT_SHAPE])
    x_val = torch.cat([x1_val, x2_val])
    print("Normalizing")
    x_val -= torch.mean(x_val)
    y_val = torch.cat([torch.zeros(5), torch.ones(5)])
    return x_train, y_train, x_val, y_val


class RandomSeparated(Dataset):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if index < self.size // 2:
            return torch.rand(INPUT_SHAPE) - 0.5, Tensor([0])
        return torch.rand(INPUT_SHAPE), Tensor([1])

    def __len__(self) -> int:
        return self.size


def test_overfit_random() -> None:
    X_TRAIN, Y_TRAIN, X_VAL, Y_VAL = random_data()
    train_loader = DataLoader(
        TensorDataset(X_TRAIN, Y_TRAIN),
        batch_size=BATCH_SIZE,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(TensorDataset(X_VAL, Y_VAL), batch_size=BATCH_SIZE, num_workers=8)
    model = MemTestCNN()
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)
    # test_loader = DataLoader(RandomSeparated(100), batch_size=BATCH_SIZE, num_workers=8)
    # trainer.test(model, test_loader)


def test_overfit_fmri_subset(is_eigimg: bool = False, preload: bool = False) -> None:
    data = FmriDataset(is_eigimg=is_eigimg, preload_data=preload)
    test_length = 40 if len(data) == 100 else 100
    train_length = len(data) - test_length
    train, val = random_split(data, (train_length, test_length), generator=None)
    print("For quick testing, subset sizes will be:")
    print(f"train: {len(train)}")
    print(f"val:   {len(val)}")
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    batch_size = args.batch_size
    if len(train) % batch_size != 0:
        warn(
            "Batch size does not evenly divide training set. "
            f"{len(train) % batch_size} subjects will be dropped each training epoch."
        )
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val,
        batch_size=40 if len(data) == 100 else BATCH_SIZE,
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )
    model = MemTestCNN()
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    seed_everything(666, workers=True)
    # test_overfit_random()
    test_overfit_fmri_subset(is_eigimg=False, preload=True)
