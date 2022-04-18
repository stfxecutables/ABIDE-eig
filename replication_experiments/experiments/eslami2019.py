# fmt: off
import logging
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
logging.getLogger("tensorflow").setLevel(logging.FATAL)
logging.getLogger("tensorboard").setLevel(logging.FATAL)
# fmt: on

from typing import Any, Tuple

import torch
from torch import Tensor
from torch.nn import Linear, ReLU, Sequential
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import accuracy, f1

from replication_experiments.models import TrainingMixin


class ASDDiagNet(TrainingMixin):
    """An attempt to replicate https://www.frontiersin.org/articles/10.3389/fninf.2019.00070/full"""

    def __init__(
        self,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        guess: float = 0.5,
        in_features: int = 9950,
        bottleneck: int = 1000,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, guess=guess, *args, **kwargs)  # type: ignore # noqa
        self.automatic_optimization = False

        self.lr = lr
        self.weight_decay = weight_decay
        self.W = Parameter(torch.randn([in_features, bottleneck]))
        self.b_enc = Parameter(torch.randn(1))
        self.b_dec = Parameter(torch.randn(1))
        self.slp = Linear(in_features=bottleneck, out_features=1, bias=True)

        self.mlp = Sequential(
            Linear(bottleneck, bottleneck // 2),
            ReLU(inplace=True),
            Linear(bottleneck // 2, 500),
            ReLU(inplace=True),
            Linear(500, 1),
        )

    def shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Tensor:
        # see https://github.com/PyTorchLightning/pytorch-lightning/issues/9806
        x, target = batch  # x.shape == (B, in_features)
        if self.global_step < 6000:
            if phase == "train":
                opt = self.optimizers()[0]  # type: ignore
                opt.zero_grad()  # type: ignore
            h_enc = torch.tanh(torch.matmul(x, self.W) + self.b_enc)
            x_prime = torch.matmul(h_enc, self.W.T) + self.b_dec
            preds = self.slp(h_enc)
            # preds = self.mlp(encoded)

            # loss = binary_cross_entropy_with_logits(preds.squeeze(), target.float())
            loss_c = binary_cross_entropy_with_logits(preds.squeeze(), target.float())
            loss_enc = mse_loss(x_prime, x)
            loss = loss_c + loss_enc
            if phase == "train":
                self.manual_backward(loss)
                opt.step()

            self.log(f"{phase}/loss", loss, prog_bar=True)
            self.log(f"{phase}/loss_c", loss_c)
            self.log(f"{phase}/loss_enc", loss_enc)
            # self.log(f"{phase}/prec", prec, prog_bar=True)
            if phase in ["val", "test"]:
                acc = accuracy(preds, target) - self.guess
                f1_score = f1(preds, target)
                self.log(f"{phase}/acc+", acc, prog_bar=True)
                self.log(f"{phase}/acc", acc + self.guess, prog_bar=True)
                self.log(f"{phase}/f1", f1_score, prog_bar=True)
            return loss
        with torch.no_grad():
            h_enc = torch.tanh(torch.matmul(x, self.W) + self.b_enc)
        if phase == "train":
            opt = self.optimizers()[1]  # type: ignore
            opt.zero_grad()  # type: ignore
        preds = self.slp(h_enc)
        loss = binary_cross_entropy_with_logits(preds.squeeze(), target.float())
        if phase == "train":
            self.manual_backward(loss)
            opt.step()

        self.log(f"{phase}/loss", loss, prog_bar=True)
        if phase in ["val", "test"]:
            acc = accuracy(preds, target) - self.guess
            f1_score = f1(preds, target)
            self.log(f"{phase}/acc+", acc, prog_bar=True)
            self.log(f"{phase}/acc", acc + self.guess, prog_bar=True)
            self.log(f"{phase}/f1", f1_score, prog_bar=True)
        return loss

    def on_train_epoch_end(self) -> None:
        if self.global_step < 6000:
            sched = self.lr_schedulers()[0]  # type: ignore
        else:
            sched = self.lr_schedulers()[1]  # type: ignore
        sched.step()

    def configure_optimizers(self) -> Any:
        # see https://github.com/PyTorchLightning/pytorch-lightning/issues/9806
        opt1 = Adam(self.parameters(), weight_decay=self.weight_decay, lr=self.lr)
        opt2 = Adam(self.slp.parameters(), weight_decay=self.weight_decay, lr=3e-4)

        # step = 500
        step = 1
        lr_decay = 0.99
        sched1 = StepLR(opt1, step_size=step, gamma=lr_decay)
        sched2 = StepLR(opt2, step_size=5, gamma=0.95)
        config1 = dict(
            optimizer=opt1,
            lr_scheduler=dict(
                scheduler=sched1,
                interval="epoch",
            ),
        )
        config2 = dict(
            optimizer=opt2,
            lr_scheduler=dict(
                scheduler=sched2,
                interval="epoch",
            ),
        )
        return config1, config2
