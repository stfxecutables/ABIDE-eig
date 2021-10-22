"""
We make tiny modifications to
https://github.com/pytorch/vision/blob/7d955df73fe0e9b47f7d6c77c699324b256fc41f/torchvision/models/resnet.py
to use Linear layers
"""

from typing import List, Tuple, no_type_check

import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import GELU, BatchNorm1d, BCEWithLogitsLoss, Conv1d, Dropout, Linear, Sequential
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
from torchmetrics.functional import accuracy, auroc
from torchmetrics.functional import f1 as f1score
from torchmetrics.functional import precision_recall


class WideTabResLayer(nn.Module):
    expansion = 4

    def __init__(
        self,
        in_features: int,
        out_features: int,
    ) -> None:
        super().__init__()
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.lin1 = Linear(in_features, out_features, bias=False)
        self.bn1 = BatchNorm1d(out_features)
        self.lin2 = Linear(out_features, out_features, bias=False)
        self.bn2 = BatchNorm1d(out_features)
        self.gelu = GELU()
        self.expand = Conv1d(in_channels=in_features, out_channels=out_features, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        out: Tensor
        identity = x
        out = self.lin1(x)
        out = self.bn1(out)
        out = self.gelu(out)
        out = self.lin2(out)
        out = self.bn2(out)
        out = self.gelu(out)

        identity = identity.unsqueeze(2)
        identity = self.expand(identity).squeeze()
        out += identity
        out = self.gelu(out)
        return out


class TabInput(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.lin = Linear(in_features, out_features, bias=False)
        self.bn = BatchNorm1d(out_features)
        self.act = GELU()

    def forward(self, x: Tensor) -> Tensor:
        x = self.lin(x)
        x = x.squeeze()
        x = self.bn(x)
        x = self.act(x)
        # x = x.unsqueeze(1)
        return x


class TabWideResNet(nn.Module):
    def __init__(
        self,
        in_features: int,
        width: int = 64,
        n_layers: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.width = width
        self.input = TabInput(in_features, width)
        layers: List[nn.Module] = []
        for d in range(n_layers):
            r_in, r_out = 2 ** d, 2 ** (d + 1)
            layers.append(WideTabResLayer(self.width * r_in, self.width * r_in))
            layers.append(WideTabResLayer(self.width * r_in, self.width * r_out))
            if dropout > 0:
                layers.append(Dropout(dropout, inplace=True))
        self.res_layers = Sequential(*layers)
        self.out = Linear(self.width * r_out, 1, bias=True)

    def forward(self, x: Tensor) -> Tensor:
        x = self.input(x)
        x = self.res_layers(x)
        x = self.out(x)
        return x


class TabLightningNet(LightningModule):
    VAL_MAX = 0.72

    def __init__(
        self,
        in_features: int,
        width: int = 64,
        n_layers: int = 4,
        dropout: float = 0.0,
        val_dummy: float = 0.5,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.in_features: int = in_features
        self.width: int = width
        self.n_layers: int = n_layers
        self.dropout = dropout
        self.val_dummy = torch.Tensor([val_dummy]).squeeze().to(device="cuda")
        self.max_gain = TabLightningNet.VAL_MAX - self.val_dummy
        self.min_gain = -self.val_dummy
        self.model = TabWideResNet(
            in_features=self.in_features,
            width=self.width,
            n_layers=self.n_layers,
            dropout=self.dropout,
        )

    # @no_type_check
    # def forward(self, x: Tensor, *args, **kwargs) -> Any:
    #     return self.model(x)

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> Tensor:
        return self.shared_step(batch, batch_idx, label="train")

    @no_type_check
    def validation_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int) -> None:
        return self.shared_step(batch, batch_idx, label="val")

    def shared_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, label: str) -> Tensor:
        x, target = batch
        pred = self.model(x).squeeze()
        criterion = BCEWithLogitsLoss()
        loss: Tensor = criterion(pred, target)
        if label != "train":
            target = target.int()
            acc = accuracy(pred, target)
            auc = auroc(pred, target)
            f1 = f1score(pred, target)
            precision, recall = precision_recall(pred, target)
            gain = acc - self.val_dummy
            # rescale gain to be in [0, 100] for goal
            goal = torch.round(1000 * gain / self.max_gain) / 1000  # round to 3 decimals
            self.log(f"{label}_acc", acc, prog_bar=True)
            self.log(f"{label}_gain", gain, prog_bar=True)
            self.log(f"{label}_goal", goal, prog_bar=True)
            self.log(f"{label}_auc", auc, prog_bar=True)
            self.log(f"{label}_f1", f1, prog_bar=True)
            self.log(f"{label}_precision", precision, prog_bar=False)
            self.log(f"{label}_recall", recall, prog_bar=False)
        self.log(f"{label}_loss", loss, prog_bar=True)
        return loss

    @no_type_check
    def configure_optimizers(self):
        # https://pytorch-lightning.readthedocs.io/en/latest/api/pytorch_lightning.core.lightning.html#pytorch_lightning.core.lightning.LightningModule.configure_optimizers
        # milestones = [50, 100, 150, 200, 300, 400, 500]
        milestones = [75, 150, 300, 500, 750, 1000]
        optimizer = Adam(self.parameters(), lr=1e-3, weight_decay=1e-5)
        # scheduler = CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
        scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=0.5)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,  # update each epoch
            },
        }


if __name__ == "__main__":
    x = torch.rand([2, 175], device="cuda")
    model = TabWideResNet(in_features=175).to("cuda")
    out = model(x)
    print(out.shape)
