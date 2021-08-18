# fmt: off
from pathlib import Path  # isort:skip
import sys  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

from argparse import ArgumentParser
from logging import warn
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import torch
from numpy import ndarray
from pandas import DataFrame, Series
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor
from torch.nn import (
    AdaptiveMaxPool3d,
    BatchNorm3d,
    BCEWithLogitsLoss,
    Conv3d,
    InstanceNorm3d,
    Linear,
    LSTMCell,
    Module,
    PReLU,
    ReLU,
)
from torch.nn.modules.padding import ConstantPad3d
from torch.utils.data import DataLoader, Dataset, random_split
from torch.utils.data.dataset import TensorDataset
from torchmetrics.functional import accuracy
from typing_extensions import Literal

from src.analysis.predict.deep_learning.constants import INPUT_SHAPE
from src.analysis.predict.deep_learning.dataloader import FmriDataset
from src.analysis.predict.deep_learning.models.layers.lstm import ConvLSTM3d
from src.analysis.predict.deep_learning.models.layers.reduce import GlobalAveragePooling
from src.analysis.predict.deep_learning.models.layers.utils import EVEN_PAD

BATCH_SIZE = 1
MODEL_ARGS = dict(
    in_channels=1,
    in_spatial_dims=INPUT_SHAPE[1:],
    num_layers=1,
    hidden_sizes=[4],
    kernel_sizes=[3],
    dilations=[2],
)

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
    def __init__(self) -> None:
        super().__init__()


class ResBlock(Module):
    def __init__(self) -> None:
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
        # BNorm  after PReLU
        # see https://github.com/ducha-aiki/caffenet-benchmark/blob/master/batchnorm.md
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
        # TODO: instead of pooling, just flatten, transpose, and treat as
        # Conv1D with e.g. 8 channels, 175 timepoints
        self.gap = GlobalAveragePooling()
        self.pool = AdaptiveMaxPool3d((1, 1, 1))
        # TODO: replace with Conv1D net
        self.linear = Linear(in_features=1400, out_features=1, bias=True)  # If no pool
        # self.linear = Linear(in_features=175, out_features=1, bias=True)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        # input shape is (B, 175, 47, 59, 42)
        x = self.conv1(x)  # now x.shape == [40, 175, 24, 30, 21]
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.conv2(x)  # now x.shape == torch.Size([40, 175, 12, 15, 11])
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.conv3(x)  # now x.shape == torch.Size([40, 175, 6, 8, 6])
        x = self.relu3(x)
        x = self.norm3(x)
        x = self.conv4(x)  # now x.shape == torch.Size([40, 175, 3, 4, 3])
        x = self.relu4(x)
        x = self.norm4(x)
        x = self.conv5(x)  # now x.shape == torch.Size([40, 175, 2, 2, 2])
        x = self.relu5(x)
        x = self.norm5(x)
        # x = self.gap(x)

        # x = self.pool(x)
        x = x.reshape([x.size(0), -1])  # flatten for if going straight to linear

        # x = x.reshape([x.size(0), 1, -1])  # flatten and add 1-channel if Conv1D
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


class Conv3dToLstm3d(LightningModule):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        IN_CH = INPUT_SHAPE[0]
        final_conv_args: Dict = dict(
            in_channels=IN_CH,
            out_channels=IN_CH,
            kernel_size=3,
            stride=2,
            dilation=3,
            padding=3,
            bias=False,
        )
        self.save_hyperparameters()
        self.model = ConvLSTM3d(**config)
        # TODO: instead of pooling, just flatten, transpose, and treat as
        # Conv1D with e.g. 8 channels, 175 timepoints
        self.padder = ConstantPad3d(EVEN_PAD, 0)
        self.conv1 = Conv3d(**final_conv_args)
        self.relu1 = PReLU()
        self.norm1 = InstanceNorm3d(IN_CH)
        self.conv2 = Conv3d(**final_conv_args)
        self.relu2 = PReLU()
        self.norm2 = InstanceNorm3d(IN_CH)
        self.conv3 = Conv3d(**final_conv_args)
        self.relu3 = PReLU()
        self.norm3 = InstanceNorm3d(IN_CH)
        # output shape after above is (1, IN_CH, 6, 8, 6)

        self.gap = GlobalAveragePooling()
        self.pool = AdaptiveMaxPool3d((1, 1, 1))
        # TODO: replace with Conv1D net
        self.linear = Linear(in_features=32, out_features=1, bias=True)  # If no pool
        # self.linear = Linear(in_features=175, out_features=1, bias=True)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        # input shape is (B, 175, 47, 59, 42)
        x, _ = self.model(x)  # now x.shape == (B, 47, 59, 42)
        x = self.padder(x)  # now x.shape == (B, 48, 60, 42)
        x = self.conv1(x)  # now x.shape == (B, 8, 24, 30, 21)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.conv2(x)  # now x.shape == torch.Size([40, 16, 12, 15, 11])
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.conv3(x)  # now x.shape == torch.Size([40, 32, 6, 8, 6])
        x = self.relu3(x)
        x = self.norm3(x)
        x = self.pool(x)  # now x.shape == torch.Size([40, 32, 1, 1, 1])
        x = x.squeeze()
        x = self.linear(x)
        # x = self.gap(x)

        # x = x.reshape([x.size(0), -1])  # flatten for if going straight to linear
        # x = x.reshape([x.size(0), 1, -1])  # flatten and add 1-channel if Conv1D
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
        loss = criterion(y_pred.squeeze(), y_true.squeeze())
        acc = accuracy(
            torch.sigmoid(y_pred.squeeze().unsqueeze(0)), y_true.squeeze().unsqueeze(0).int()
        )
        return acc, loss


class Lstm3dToConv(LightningModule):
    def __init__(self, config: Dict) -> None:
        super().__init__()
        final_conv_args: Dict = dict(
            kernel_size=3,
            stride=2,
            dilation=3,
            padding=3,
            bias=False,
        )
        self.save_hyperparameters()
        self.model = ConvLSTM3d(**config)
        # TODO: instead of pooling, just flatten, transpose, and treat as
        # Conv1D with e.g. 8 channels, 175 timepoints
        self.padder = ConstantPad3d(EVEN_PAD, 0)
        self.conv1 = Conv3d(
            in_channels=self.model.hidden_sizes[-1], out_channels=8, **final_conv_args
        )
        self.relu1 = PReLU()
        self.norm1 = InstanceNorm3d(8)
        self.conv2 = Conv3d(in_channels=8, out_channels=16, **final_conv_args)
        self.relu2 = PReLU()
        self.norm2 = InstanceNorm3d(16)
        self.conv3 = Conv3d(in_channels=16, out_channels=32, **final_conv_args)
        self.relu3 = PReLU()
        self.norm3 = InstanceNorm3d(32)

        self.gap = GlobalAveragePooling()
        self.pool = AdaptiveMaxPool3d((1, 1, 1))
        # TODO: replace with Conv1D net
        self.linear = Linear(in_features=32, out_features=1, bias=True)  # If no pool
        # self.linear = Linear(in_features=175, out_features=1, bias=True)

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        # input shape is (B, 175, 47, 59, 42)
        x, _ = self.model(x)  # now x.shape == (B, 47, 59, 42)
        x = self.padder(x)  # now x.shape == (B, 48, 60, 42)
        x = self.conv1(x)  # now x.shape == (B, 8, 24, 30, 21)
        x = self.relu1(x)
        x = self.norm1(x)
        x = self.conv2(x)  # now x.shape == torch.Size([40, 16, 12, 15, 11])
        x = self.relu2(x)
        x = self.norm2(x)
        x = self.conv3(x)  # now x.shape == torch.Size([40, 32, 6, 8, 6])
        x = self.relu3(x)
        x = self.norm3(x)
        x = self.pool(x)  # now x.shape == torch.Size([40, 32, 1, 1, 1])
        x = x.squeeze()
        x = self.linear(x)
        # x = self.gap(x)

        # x = x.reshape([x.size(0), -1])  # flatten for if going straight to linear
        # x = x.reshape([x.size(0), 1, -1])  # flatten and add 1-channel if Conv1D
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
        loss = criterion(y_pred.squeeze(), y_true.squeeze())
        acc = accuracy(
            torch.sigmoid(y_pred.squeeze().unsqueeze(0)), y_true.squeeze().unsqueeze(0).int()
        )
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
    model = Lstm3dToConv(config=MODEL_ARGS)
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.default_root_dir = ROOT / "results/LSTM_rand_test"
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)
    # test_loader = DataLoader(RandomSeparated(100), batch_size=BATCH_SIZE, num_workers=8)
    # trainer.test(model, test_loader)


def test_overfit_fmri_subset(is_eigimg: bool = False, preload: bool = False) -> None:
    data = FmriDataset(is_eigimg=is_eigimg, preload_data=preload)
    test_length = 40 if len(data) == 100 else 100
    train_length = len(data) - test_length
    train, val = random_split(data, (train_length, test_length), generator=None)
    val_aut = torch.cat(list(zip(*list(val)))[1]).sum().int().item()
    train_aut = torch.cat(list(zip(*list(train)))[1]).sum().int().item()
    print("For quick testing, subset sizes will be:")
    print(f"train: {len(train)} (Autism={train_aut}, Control={len(train) - train_aut})")
    print(f"val:   {len(val)} (Autism={val_aut}, Control={len(val) - val_aut})")
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
    seed_everything(333, workers=True)
    test_overfit_random()
    # test_overfit_fmri_subset(is_eigimg=True, preload=True)
