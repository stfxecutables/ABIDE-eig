from __future__ import annotations

from argparse import Namespace
from copy import deepcopy
from datetime import timedelta
from typing import Any, Dict, List, Optional

import numpy as np
from optuna import Trial, TrialPruned
from pl_bolts.callbacks import TrainingDataMonitor
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback, GPUStatsMonitor, ModelCheckpoint


# see https://forums.pytorchlightning.ai/t/how-to-access-the-logged-results-such-as-losses/155/8
class OptunaHelper(Callback):
    def __init__(self, trial: Trial, pruning_metric: str = "val_loss", smooth: int = 3) -> None:
        super().__init__()
        self.metrics = {}
        self.trial = trial
        self.val_step = -1
        self.pruning_metric = pruning_metric
        self.smooth = smooth

    def on_validation_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        self.val_step += 1

    def on_validation_epoch_end(self, trainer: Trainer, pl_module: LightningModule) -> None:
        metrics = deepcopy(trainer.callback_metrics)
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = []
                self.metrics[name].append(value.item())
        # `metrics` is now a dict with whatever metric keys defined in the LightningModule
        # e.g. 'train_acc', 'train_acc_step', 'loss', 'loss_step', 'val_loss', 'val_acc'
        if self.pruning_metric not in self.metrics:  # e.g. val sanity check
            return
        # get last `smooth` metric values
        smoothed_metric = np.mean(self.metrics[self.pruning_metric][-self.smooth :])
        self.trial.report(smoothed_metric, self.val_step)
        if self.trial.should_prune():
            raise TrialPruned()


def callbacks(config: Namespace, trial: Trial) -> List[Callback]:
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
        ),
        OptunaHelper(trial),
    ]
    return list(filter(lambda c: c is not None, cbs))  # type: ignore
