from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple, no_type_check

import torch
from joblib import Memory
from pytorch_lightning import LightningModule
from torch import Tensor
from torch.nn import (
    BatchNorm1d,
    BatchNorm2d,
    Conv1d,
    Conv2d,
    Dropout,
    LazyLinear,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
)
from torch.nn.functional import binary_cross_entropy_with_logits, mse_loss
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchmetrics.functional import accuracy, f1

ROOT = Path(__file__).resolve().parent
LOGS = ROOT / "lightning_logs"
MEMOIZER = Memory("__CACHE__")

global GUESS
GUESS = None


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

    @no_type_check
    def training_step(self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs) -> Any:
        return self.shared_step(batch, "train")

    @no_type_check
    def validation_step(
        self, batch: Tuple[Tensor, Tensor], batch_idx: int, *args, **kwargs
    ) -> Optional[Any]:
        self.shared_step(batch, "val")

    @no_type_check
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
            # Conv(in_channels=6, out_channels=init_ch, kernel_size=N),
            # Flatten(),
            LazyLinear(out_features=init_ch, bias=True),
            # BatchNorm1d(init_ch),
            LeakyReLU(inplace=True),
            Dropout(p=dropout)
            # LazyLinear(out_features=init_ch, bias=True),
            # LeakyReLU(inplace=True),
        ]
        ch = init_ch
        out = ch
        for _ in range(depth - 1):
            out = min(max_channels, out * 2)
            # out = max(1, out // 2)
            # layers.append(CorrCell(ch, out))
            layers.append(Linear(ch, out))
            # layers.append(BatchNorm1d(out))
            layers.append(LeakyReLU(inplace=True))
            # layers.append(Dropout(p=dropout))
            ch = out
        # will have shape (B, out, LEN)
        # layers.append(GlobalAveragePool2D())
        layers.append(Linear(out, 1, bias=True))
        self.model = Sequential(*layers)

    def shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Tensor:
        """
        channel sds:
        0  0.105912
        1  0.067476
        2  0.077311
        3  0.076805
        4  0.076704
        5  0.077211
        """
        sds = [0.105912, 0.067476, 0.077311, 0.076805, 0.076704, 0.077211]
        x, target = batch
        errs = torch.stack(
            [torch.normal(mean=0.0, std=sd / 3, size=(x.shape[0], *x.shape[2:])) for sd in sds],
            dim=1,
        )

        x = x + errs.to(x.device)
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


class ASDDiagNet(TrainingMixin):
    """An attempt to replicate https://www.frontiersin.org/articles/10.3389/fninf.2019.00070/full"""

    def __init__(
        self,
        in_features: int = 9950,
        bottleneck: int = 1000,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
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
        x, target = batch  # x.shape == (B, in_features)
        h_enc = torch.tanh(torch.matmul(x, self.W) + self.b_enc)
        x_prime = torch.matmul(h_enc, self.W.T) + self.b_dec
        preds = self.slp(h_enc)
        # preds = self.mlp(encoded)

        # loss = binary_cross_entropy_with_logits(preds.squeeze(), target.float())
        loss_c = binary_cross_entropy_with_logits(preds.squeeze(), target.float())
        loss_enc = mse_loss(x_prime, x)
        loss = loss_c + loss_enc

        self.log(f"{phase}/loss", loss)
        self.log(f"{phase}/loss_c", loss_c)
        self.log(f"{phase}/loss_enc", loss_enc)
        # self.log(f"{phase}/prec", prec, prog_bar=True)
        if phase in ["val", "test"]:
            acc = accuracy(preds, target) - GUESS
            f1_score = f1(preds, target)
            self.log(f"{phase}/acc+", acc, prog_bar=True)
            self.log(f"{phase}/acc", acc + GUESS, prog_bar=True)
            self.log(f"{phase}/f1", f1_score, prog_bar=True)
        return loss


class SharedAutoEncoder(TrainingMixin):
    def __init__(
        self,
        in_features: int,
        depth: int = 2,
        bottleneck: int = 1000,
        lr: float = 3e-4,
        weight_decay: float = 0.0,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.lr = lr
        self.weight_decay = weight_decay
        if depth == 1:
            self.encoder = Sequential(Linear(in_features, bottleneck), ReLU(inplace=True))
            self.decoder = Sequential(Linear(bottleneck, in_features), ReLU(inplace=True))
        else:
            enc_layers, dec_layers = [], []
            ch = in_features
            for d in range(depth):
                if d != depth - 1:
                    ch2 = max(1, ch // 2)
                    enc_layers.append(Linear(ch, ch2))
                    enc_layers.append(ReLU())
                    dec_layers.append(Linear(ch2, ch))
                    dec_layers.append(ReLU())
                    ch = max(1, ch // 2)
                else:
                    enc_layers.append(Linear(ch, bottleneck))
                    enc_layers.append(ReLU())
                    dec_layers.append(Linear(bottleneck, ch))
                    dec_layers.append(ReLU())

            self.encoder = Sequential(*enc_layers)
            self.decoder = Sequential(*reversed(dec_layers))
        # self.decoder = Identity()
        print(self.encoder)
        print(self.decoder)
        # self.linear = Linear(bottleneck, 1, bias=True)
        self.mlp = Sequential(
            Linear(bottleneck, bottleneck // 2),
            ReLU(inplace=True),
            Linear(bottleneck // 2, 500),
            ReLU(inplace=True),
            Linear(500, 1),
        )
        # self.relu = ReLU()

    def shared_step(self, batch: Tuple[Tensor, Tensor], phase: str) -> Tensor:
        """
        channel sds:
        0  0.105912
        1  0.067476
        2  0.077311
        3  0.076805
        4  0.076704
        5  0.077211
        """
        # sds = [0.105912, 0.067476, 0.077311, 0.076805, 0.076704, 0.077211]
        # sds = [0.105912, 0.067476]
        x, target = batch
        x = x[:, 0]
        # errs = torch.stack(
        #     [torch.normal(mean=0.0, std=sd / 3, size=(x.shape[0], *x.shape[2:])) for sd in sds],
        #     dim=1,
        # )

        # x = x + errs.to(x.device)
        x = torch.flatten(x, 1)  # x.shape == (B, in_features)
        # encoded = self.relu(torch.matmul(x, self.W))  # (B, bottlenck)
        # decoded = torch.matmul(encoded, self.W.T)
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)

        preds = self.mlp(encoded)
        # loss = binary_cross_entropy_with_logits(preds.squeeze(), target.float())
        loss_c = binary_cross_entropy_with_logits(preds.squeeze(), target.float())
        loss_enc = mse_loss(x, decoded)
        loss = loss_c + loss_enc

        self.log(f"{phase}/loss", loss)
        # self.log(f"{phase}/prec", prec, prog_bar=True)
        if phase in ["val", "test"]:
            acc = accuracy(preds, target) - GUESS
            f1_score = f1(preds, target)
            self.log(f"{phase}/acc+", acc, prog_bar=True)
            self.log(f"{phase}/acc", acc + GUESS, prog_bar=True)
            self.log(f"{phase}/f1", f1_score, prog_bar=True)
        return loss
