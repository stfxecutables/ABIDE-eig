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

import re
import traceback
from random import shuffle
from typing import Any, Dict, List

import pandas as pd
from pandas import DataFrame
from pytorch_lightning import LightningModule
from scipy.stats.distributions import binom, expon, loguniform, randint, uniform
from sklearn.model_selection import ParameterGrid, ParameterSampler
from torch.nn import Linear, Module, Sequential
from yaml import UnsafeLoader, load, safe_load

from replication_experiments.constants import Norm
from replication_experiments.evaluation import LOGS, test_split
from replication_experiments.layers import SoftLinear
from replication_experiments.models import TrainingMixin

GRID_LOGDIR = f"SeqLengths--Grid"

# hparams that NEED to be replicated
N = int((200 ** 2 - 200) / 2)  # upper triangle of 200x200 matrix where diagonals are 1
BATCH_SIZE = 16
LR = 6e-4

# options that should be irrelevant, if set correctly (e.g. large enough)
FEAT_SELECT = "sd"  # no effect with n == N
NORM: Norm = "feature"  # this is the only one that works
MAX_STEPS = 10_000
MAX_EPOCHS = 500
TRAINER_ARGS = dict(max_epochs=MAX_EPOCHS, max_steps=MAX_STEPS)


class SoftLinearModel(TrainingMixin):
    def __init__(
        self,
        in_features: int = 19900,
        init_ch: int = 16,
        depth: int = 4,
        max_channels: int = 512,
        dropout: float = 0.0,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        guess: float = 0.5,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, guess=guess, *args, **kwargs)  # type: ignore # noqa
        self.save_hyperparameters()
        layers: List[Module] = [SoftLinear(in_features, init_ch, dropout)]
        ch = init_ch
        out = ch
        for _ in range(depth - 1):
            out = min(max_channels, out * 2)
            layers.append(SoftLinear(ch, out))
            ch = out
        layers.append(Linear(out, 1, bias=True))
        self.model = Sequential(*layers)


def test_exhaustive_grid() -> None:
    """
    Notes
    -----
    # Findings
    If we condition on high val_acc, we discover some things:

    - there are more paths through max_channels=16 and 64 than through 256
      - but not when in_features is > N // 4
    - there are more paths through depth=2, 4 than depth=8
    - there are more paths through init_ch=64 than lower values
    - there are more paths through in_features > N//4
    - with above depth/breadth factors, dropout should be < 0.5, usually more like 0-0.1
    - high val acc can only be obtained when there is not too much regulartization.
      That is, when dropout in [0.3, 1.0] and wd in [3e-3, 1e-1], you can get acc+ >4,
      but when
    - strong weight decay (>1e-3) strongly conditions what other hparams can work:
    """
    # TODO: Switch to random grid
    grid = list(
        ParameterSampler(
            dict(
                in_features=[1],
                init_ch=[4, 16, 64],
                depth=[2, 4, 8],
                max_channels=[16, 64, 256],
                dropout=uniform(loc=0, scale=1),
                weight_decay=loguniform(1e-6, 1e-1),
                lr=loguniform(1e-5, 1e-2),
            ),
            n_iter=300,
        )
    )
    # grid = list(
    #     ParameterGrid(
    #         dict(
    #             in_features=[N // 2, N // 4, 2000],
    #             init_ch=[16, 64],
    #             depth=[2, 4, 8],
    #             max_channels=[64, 256],
    #             dropout=[0.0, 0.25, 0.5, 0.75],
    #             weight_decay=[0.0, 1e-4],
    #             lr=[3e-4, 6e-4],
    #         )
    #     )
    # )
    # shuffle(grid)

    for i, model_args in enumerate(grid):
        print(f"Iteration {i} of {len(grid)}.")
        print(f"Testing model with args: {model_args}")
        try:
            test_split(
                n_features=model_args["in_features"],
                feat_select=FEAT_SELECT,
                norm=NORM,
                batch_size=BATCH_SIZE,
                model_cls=SoftLinearModel,
                model_args=model_args,
                trainer_args={**TRAINER_ARGS, **dict(enable_progress_bar=False)},
                logdirname=GRID_LOGDIR,
                source="lengths",
            )
        except Exception:
            traceback.print_exc()


def get_results_table(logdirname: str = GRID_LOGDIR) -> DataFrame:
    def load_yaml(path: Path) -> Any:
        with open(path, "r") as handle:
            return load(handle, Loader=UnsafeLoader)

    bests = sorted((LOGS / logdirname).rglob("*.ckpt"))
    get_val = lambda p: float(re.search(r"val_acc=(.*)", p.stem).group(1))  # type: ignore
    best_ckpts = list(reversed(sorted(bests, key=get_val)))
    accs = list(reversed(sorted([get_val(p) for p in bests])))
    best_yamls = [p.parent.parent / "hparams.yaml" for p in best_ckpts]
    best_params = [load_yaml(path) for path in best_yamls]
    keys = set()
    params: Dict
    rows = []
    for acc, params in zip(accs, best_params):
        rows.append(DataFrame({**dict(val_acc=acc), **params}, index=[params["uuid"]]))
        for key in params.keys():
            keys.add(key)
    df = pd.concat(rows, axis=0, ignore_index=False)
    df.drop(columns="uuid")
    pd.options.display.max_rows = 1000
    print(df)
    df.to_parquet(ROOT / "mlp_fits.parquet")
    print(f"df saved to {ROOT / 'mlp_fits.parquet'}")


def test_grid_best(logdirname: str = GRID_LOGDIR) -> None:
    REPEATS = 1

    def load_yaml(path: Path) -> Any:
        with open(path, "r") as handle:
            return load(handle, Loader=UnsafeLoader)

    bests = sorted((LOGS / logdirname).rglob("*.ckpt"))
    get_val = lambda p: float(re.search(r"val_acc=(.*)", p.stem).group(1))  # type: ignore
    best_ckpts = list(reversed(sorted(bests, key=get_val)))
    best_yamls = [p.parent.parent / "hparams.yaml" for p in best_ckpts[:5]]
    best_params = [load_yaml(path) for path in best_yamls]
    for params in best_params:
        params.pop("uuid")  # will cause conflicts in running new models
    print(best_params)

    for i, model_args in enumerate(best_params):
        print(f"Parameter combination {i} of {len(best_params)}.")
        print(f"Testing model with args: {model_args}")
        for _ in range(REPEATS):
            try:
                test_split(
                    n_features=model_args["in_features"],
                    feat_select=FEAT_SELECT,
                    norm=NORM,
                    batch_size=BATCH_SIZE,
                    model_cls=SoftLinearModel,
                    model_args=model_args,
                    trainer_args=TRAINER_ARGS,
                    logdirname=GRID_LOGDIR.replace("Grid", "Best"),
                    source="lengths",
                )
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    test_exhaustive_grid()
    # test_grid_best()
    # get_results_table()
