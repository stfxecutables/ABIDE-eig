from __future__ import annotations  # noqa

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
from dataclasses import dataclass
from math import prod
from typing import Any, Dict, Tuple, Type, no_type_check

from pytorch_lightning import LightningModule, Trainer, seed_everything
from torch import Tensor
from torch.nn import BCEWithLogitsLoss, Flatten, Linear, ModuleList
from torch.nn.modules.padding import ConstantPad3d
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torchmetrics.functional import accuracy

from src.analysis.predict.deep_learning.arguments import get_args
from src.analysis.predict.deep_learning.dataloader import FmriDataset
from src.analysis.predict.deep_learning.models.layers.conv import MultiConv4D
from src.analysis.predict.deep_learning.tables import tableify_logs
from src.constants.shapes import EVEN_PAD, FMRI_PADDED_SHAPE


@dataclass
class MultiNetConfig:
    num_layers: int = 5
    channel_expansion: int = 2
    channel_exp_start: int = 4
    spatial_kernel: int = 3
    spatial_stride: int = 2
    spatial_dilation: int = 2
    temporal_kernel: int = 2
    temporal_stride: int = 2
    temporal_dilation: int = 1
    spatial_padding: int = 2
    temporal_padding: int = 0
    lr: float = 1e-4
    l2: float = 1e-5

    @property
    def namespace(self) -> Namespace:
        return Namespace(**self.__dict__)

    @property
    def kwargs(self) -> Dict[str, Any]:
        """Just get kwargs for MultiNet"""
        d = deepcopy(self.__dict__)  # copy
        d.pop("num_layers")
        d.pop("channel_exp_start")
        d.pop("channel_expansion")
        d.pop("lr")
        d.pop("l2")
        return d

    @staticmethod
    def default() -> MultiNetConfig:
        return MultiNetConfig()


"""
INCREASE THE SPATIAL STRIDE SO THAT WE GET DOWNSAMPLING, AND SEE THE MEMORY
COSTS THEN!

"""


class MultiNet(LightningModule):
    def __init__(self, config: MultiNetConfig = MultiNetConfig.default()) -> None:
        super().__init__()
        self.save_hyperparameters(config.namespace)
        self.config = deepcopy(config)
        # for i in range(self.hparams.num_layers):

        self.layers = ModuleList([ConstantPad3d(EVEN_PAD, 0)])
        s_in = FMRI_PADDED_SHAPE[1:]
        t_in = FMRI_PADDED_SHAPE[0]
        in_ch, exp = 1, self.config.channel_exp_start
        for i in range(self.config.num_layers):
            conv = MultiConv4D(
                in_channels=in_ch,
                channel_expansion=exp,
                spatial_in_shape=s_in,
                temporal_in_shape=t_in,
                **self.config.kwargs,
            )
            s_in = conv.spatial_outshape()
            t_in = conv.temporal_outshape()
            for s in s_in:
                if s <= 0:
                    raise ValueError(
                        f"Spatial feature map size has shrunk to zero after layer {i}."
                    )
            in_ch *= exp
            exp = self.config.channel_expansion  # no longer use starting expansion
            self.layers.append(conv)
        lin_ch = in_ch * t_in * prod(s_in)
        self.layers.append(Flatten())
        self.layers.append(Linear(lin_ch, 1, bias=True))
        print(self)

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
        lr, wd = self.config.lr, self.config.l2
        return Adam(self.parameters(), lr=lr, weight_decay=wd)

    def shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Tuple[Tensor, Tensor]:
        loss: Tensor
        x, target = batch
        x = x.unsqueeze(1)
        preds = self(x)
        criterion = BCEWithLogitsLoss()
        loss = criterion(preds.squeeze(), target.squeeze())
        acc = accuracy(preds, target.int())
        if phase != "train":
            self.log(f"{phase}_loss", loss, prog_bar=True)
        self.log(f"{phase}_acc", acc, prog_bar=True)
        return acc, loss

    def __str__(self) -> str:
        lines = []
        for layer in self.layers:
            lines.append(str(layer))
        return "\n".join(lines)


def train_model(
    model_class: Type,
) -> None:
    args = get_args(model_class)
    # config = model_class.config(args)
    config = MultiNetConfig.default()
    data = FmriDataset(args)
    train, val = data.train_val_split(args)
    model = model_class(config)
    # trainer = Trainer.from_argparse_args(args, callbacks=callbacks(config))
    trainer = Trainer.from_argparse_args(args)
    trainer.logger.log_hyperparams(config.namespace)
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
