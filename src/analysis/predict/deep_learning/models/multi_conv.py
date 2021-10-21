# fmt: off
from pathlib import Path  # isort:skip
import sys  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

from argparse import Namespace
from copy import deepcopy
from math import prod
from typing import Any, Dict, Tuple, Type, no_type_check

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor
from torch.nn import (
    AdaptiveMaxPool3d,
    BCEWithLogitsLoss,
    Conv3d,
    Flatten,
    InstanceNorm3d,
    Linear,
    Module,
    ModuleList,
    PReLU,
    Sequential,
)
from torch.nn.modules.padding import ConstantPad3d
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

from src.analysis.predict.deep_learning.arguments import get_args
from src.analysis.predict.deep_learning.callbacks import callbacks
from src.analysis.predict.deep_learning.constants import INPUT_SHAPE, PADDED_SHAPE
from src.analysis.predict.deep_learning.dataloader import FmriDataset
from src.analysis.predict.deep_learning.models.layers.conv import MultiConv4D
from src.analysis.predict.deep_learning.models.layers.reduce import GlobalAveragePooling
from src.analysis.predict.deep_learning.models.layers.utils import EVEN_PAD
from src.analysis.predict.deep_learning.tables import tableify_logs

DEFAULTS = Namespace(
    **dict(
        num_layers=4,
        channel_expansion=2,
        channel_exp_start=2,
        spatial_kernel=3,
        spatial_stride=1,
        spatial_dilation=1,
        temporal_kernel=5,
        temporal_stride=2,
        temporal_dilation=1,
        spatial_padding="same",
        temporal_padding=0,
    )
)


class MultiNet(LightningModule):
    def __init__(self, config: Namespace = DEFAULTS) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        self.config = deepcopy(config)
        self.config.__dict__.pop("num_layers")
        self.config.__dict__.pop("channel_expansion")
        # for i in range(self.hparams.num_layers):

        self.layers = ModuleList([ConstantPad3d(EVEN_PAD, 0)])
        s_in = PADDED_SHAPE[1:]
        t_in = PADDED_SHAPE[0]
        in_ch, exp = 1, self.config.__dict__.pop("channel_exp_start")
        for i in range(self.hparams.num_layers):
            conv = MultiConv4D(
                in_channels=in_ch,
                channel_expansion=exp,
                spatial_in_shape=s_in,
                temporal_in_shape=t_in,
                **self.config.__dict__,
            )
            s_in = conv.spatial_outshape()
            t_in = conv.temporal_outshape()
            in_ch *= exp
            exp = self.hparams.channel_expansion  # no longer use starting expansion
            self.layers.append(conv)
        lin_ch = in_ch * t_in * prod(s_in)
        self.layers.append(Flatten())
        self.layers.append(Linear(lin_ch, 1, bias=True))

    @no_type_check
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        acc, loss = self.shared_step(batch, "train")
        return loss

    @no_type_check
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        self.shared_step(batch, "val")

    @no_type_check
    def test_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        self.shared_step(batch, "test")

    def configure_optimizers(self) -> Any:
        # lr, wd = self.hparams.lr, self.hparams.l2  # type: ignore
        return Adam(self.parameters(), lr=1e-4, weight_decay=1e-5)

    def shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Tuple[Tensor, Tensor]:
        loss: Tensor
        x, target = batch
        x = x.unsqueeze(1)
        preds = self(x)
        criterion = BCEWithLogitsLoss()
        loss = criterion(preds.squeeze(), target.squeeze())
        acc = accuracy(preds, target.int())
        self.log(f"{phase}_loss", loss, prog_bar=True)
        self.log(f"{phase}_acc", acc, prog_bar=True)
        return acc, loss


def train_model(
    model_class: Type,
) -> None:
    args = get_args(model_class)
    # config = model_class.config(args)
    config = DEFAULTS
    data = FmriDataset(args)
    train, val = data.train_val_split(args)
    model = model_class(config)
    # trainer = Trainer.from_argparse_args(args, callbacks=callbacks(config))
    trainer = Trainer.from_argparse_args(args)
    trainer.logger.log_hyperparams(config)
    train_loader = DataLoader(
        train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val,
        batch_size=args.val_batch_size,
        num_workers=args.num_workers,
        shuffle=False,
        drop_last=False,
    )
    trainer.fit(model, train_loader, val_loader)
    tableify_logs(trainer)


if __name__ == "__main__":
    seed_everything(333, workers=True)
    train_model(MultiNet)
