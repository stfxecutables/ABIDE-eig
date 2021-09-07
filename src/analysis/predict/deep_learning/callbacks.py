from __future__ import annotations

from argparse import Namespace
from datetime import timedelta
from typing import Dict, List

from pl_bolts.callbacks import TrainingDataMonitor
from pytorch_lightning.callbacks import Callback, GPUStatsMonitor, ModelCheckpoint


def callbacks(config: Namespace) -> List[Callback]:
    ckpt_args: Dict = dict(
        auto_insert_metric_name=True,
        save_last=False,
        save_top_k=2,
        save_weights_only=False,
        train_time_interval=timedelta(minutes=5),
    )
    cbs = [
        # LearningRateMonitor(logging_interval="epoch") if config["lr_schedule"] else None,
        # EarlyStopping("val_acc", min_delta=0.001, patience=100, mode="max"),
        ModelCheckpoint(
            filename="{epoch}-{step}_{val_acc:.2f}_{train_acc:0.3f}",
            monitor="val_acc",
            mode="max",
            **ckpt_args
        ),
        ModelCheckpoint(
            filename="{epoch}-{step}_{val_loss:.2f}_{train_acc:0.3f}",
            monitor="val_loss",
            mode="min",
            **ckpt_args
        ),
        GPUStatsMonitor(
            memory_utilization=True,
            gpu_utilization=True,
            intra_step_time=True,
            inter_step_time=True,
            fan_speed=False,
            temperature=False,
        )
    ]
    return list(filter(lambda c: c is not None, cbs))  # type: ignore
