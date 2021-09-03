# fmt: off
from pathlib import Path  # isort:skip
import sys  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

from argparse import ArgumentParser, Namespace
from logging import warn
from typing import Any, Dict, Tuple, Type, no_type_check

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.profiler import AdvancedProfiler
from torch import Tensor
from torch.nn import (
    AdaptiveMaxPool3d,
    BCEWithLogitsLoss,
    Conv3d,
    InstanceNorm3d,
    Linear,
    ModuleList,
    PReLU,
)
from torch.nn.modules.padding import ConstantPad3d
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy

from src.analysis.predict.deep_learning.constants import INPUT_SHAPE, PADDED_SHAPE
from src.analysis.predict.deep_learning.dataloader import FmriDataset
from src.analysis.predict.deep_learning.models.layers.conv import ResBlock3d
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


# based on
# https://github.com/Hzzone/Precipitation-Nowcasting/blob/master/nowcasting/models/convLSTM.py
# https://github.com/SreenivasVRao/ConvGRU-ConvLSTM-PyTorch/blob/master/convlstm.py
# https://github.com/ndrplz/ConvLSTM_pytorch/blob/master/convlstm.py


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


class Conv3dToConvLstm3d(LightningModule):
    def __init__(self, config: Namespace) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        res_block_args: Dict = {
            key.replace("conv_", ""): val
            for key, val in config.__dict__.items()
            if key.startswith("conv")
        }
        lstm_args: Dict = {
            key.replace("lstm_", ""): val
            for key, val in config.__dict__.items()
            if key.startswith("lstm")
        }
        self.conv_num_layers = res_block_args.pop("num_layers")
        self.padder = ConstantPad3d(EVEN_PAD, 0)
        self.convs = ModuleList([ResBlock3d(**res_block_args) for _ in range(self.conv_num_layers)])

        spatial_out = PADDED_SHAPE[1:]
        for conv in self.convs:
            spatial_out = conv.output_shape(spatial_out)
        self.spatial_out = spatial_out
        self.conv_lstm = ConvLSTM3d(
            **lstm_args,
            in_spatial_dims=spatial_out,
            min_gpu=False,
        )
        self.linear = Linear(
            in_features=self.hparams.lstm_hidden_sizes[-1] * np.prod(spatial_out),  # type: ignore
            out_features=1,
            bias=True,
        )

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        # input shape is (B, 175, 47, 59, 42)
        # x.shape == (B, 175, 48, 60, 42) after padding, then convs do e.g.
        # x.shape == (B, 175, 24, 30, 21)
        # x.shape == (B, 175, 12, 15, 10)
        # x.shape == (B, 175, 6, 7, 5)
        x = self.padder(x)
        for conv in self.convs:
            x = conv(x)
        x = x.unsqueeze(2)
        x, _ = self.conv_lstm(x)
        x = x.reshape((x.size(0), -1))
        x = self.linear(x)
        return x

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        acc, loss = self.inner_step(batch)
        self.log("train_acc", acc, prog_bar=True, on_epoch=True, on_step=True)
        self.log("loss", loss, prog_bar=True, on_epoch=True, on_step=True)
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
        lr, wd = self.hparams.lr, self.hparams.l2  # type: ignore
        return Adam(self.parameters(), lr=lr, weight_decay=wd)

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


def test_convlstm(
    model_class: Type,
    config: Namespace,
    preload: bool = False,
    profile: bool = False,
) -> None:
    import logging

    logging.basicConfig(level=logging.INFO)
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=BATCH_SIZE, type=int)
    parser.add_argument("--is_eigimg", action="store_true")
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    is_eigimg = args.is_eigimg
    root_dir = ROOT / f"lightning_logs/{model_class.__name__}/{'eigimg' if is_eigimg else 'func'}"
    batch_size = args.batch_size
    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html?highlight=logging#logging-frequency
    args.log_every_n_steps = 5
    args.flush_logs_every_n_steps = 20
    args.default_root_dir = root_dir

    data = FmriDataset(is_eigimg=is_eigimg, preload_data=preload)
    test_length = 40 if len(data) == 100 else 100
    train_length = len(data) - test_length
    train, val = random_split(data, (train_length, test_length), generator=None)
    val_aut = torch.cat(list(zip(*list(val)))[1]).sum().int().item()  # type: ignore
    train_aut = torch.cat(list(zip(*list(train)))[1]).sum().int().item()  # type: ignore
    print("For quick testing, subset sizes will be:")
    print(f"train: {len(train)} (Autism={train_aut}, Control={len(train) - train_aut})")
    print(f"val:   {len(val)} (Autism={val_aut}, Control={len(val) - val_aut})")

    if profile:
        profiler = AdvancedProfiler(dirpath=None, filename="profiling", line_count_restriction=2.0)
        args.profiler = profiler
    # args.default_root_dir =
    if len(train) % batch_size != 0:
        warn(
            "Batch size does not evenly divide training set. "
            f"{len(train) % batch_size} subjects will be dropped each training epoch."
        )
    model = model_class(config)
    trainer = Trainer.from_argparse_args(args)
    trainer.logger.log_hyperparams(config)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        num_workers=8,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val,
        batch_size=2,
        num_workers=8,
        shuffle=False,
        drop_last=False,
    )
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    seed_everything(333, workers=True)
    config = Namespace(
        **dict(
            conv_in_channels=INPUT_SHAPE[0],
            conv_out_channels=INPUT_SHAPE[0],
            conv_num_layers=4,
            conv_kernel_size=3,
            conv_dilation=1,
            conv_residual=True,
            conv_halve=True,
            conv_depthwise=True,
            conv_depthwise_factor=None,
            conv_norm="group",
            conv_norm_groups=5,
            lstm_in_channels=1,
            lstm_num_layers=1,
            lstm_hidden_sizes=[32],
            lstm_kernel_sizes=[3],
            lstm_dilations=[1],
            lstm_norm="group",
            lstm_norm_groups=16,
            lstm_inner_spatial_dropout=0.4,
            lr=1e-4,
            l2=1e-5,
        )
    )
    test_convlstm(Conv3dToConvLstm3d, config, preload=True, profile=False)
