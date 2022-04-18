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
from typing import Any, Dict, List

from sklearn.model_selection import ParameterGrid
from torch.nn import Linear, Module, Sequential
from yaml import safe_load

from replication_experiments.constants import Norm
from replication_experiments.evaluation import LOGS, test_split
from replication_experiments.layers import Lin
from replication_experiments.models import TrainingMixin

GRID_LOGDIR = "SimpleLinear--Grid"


class LinearModel(TrainingMixin):
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
        layers: List[Module] = [Lin(in_features, init_ch, dropout)]
        ch = init_ch
        out = ch
        for _ in range(depth - 1):
            out = min(max_channels, out * 2)
            layers.append(Lin(ch, out))
            ch = out
        layers.append(Linear(out, 1, bias=True))
        self.model = Sequential(*layers)


def test_exhaustive_grid() -> None:
    # hparams that NEED to be replicated
    N = int((200 ** 2 - 200) / 2)  # upper triangle of 200x200 matrix where diagonals are 1
    BATCH_SIZE = 16
    LR = 3e-4
    FEAT_SELECT = "sd"  # no effect with n == N
    NORM: Norm = "feature"  # this is the only one that works

    # options that should be irrelevant, if set correctly (e.g. large enough)
    MAX_STEPS = 3000
    MAX_EPOCHS = 150
    TRAINER_ARGS = dict(max_epochs=MAX_EPOCHS, max_steps=MAX_STEPS)

    # test all linear models
    grid = list(
        ParameterGrid(
            dict(
                in_features=[N // 2, N // 4, 2000],
                init_ch=[16, 64],
                depth=[2, 4, 8],
                max_channels=[64, 256],
                dropout=[0.0, 0.25, 0.5, 0.75],
                weight_decay=[0.0, 1e-4],
                lr=[LR],
            )
        )
    )

    for i, model_args in enumerate(grid):
        print(f"Iteration {i} of {len(grid)}.")
        print(f"Testing model with args: {model_args}")
        try:
            test_split(
                n_features=model_args["in_features"],
                feat_select=FEAT_SELECT,
                norm=NORM,
                batch_size=BATCH_SIZE,
                model_cls=LinearModel,
                model_args=model_args,
                trainer_args=TRAINER_ARGS,
                logdirname=GRID_LOGDIR,
            )
        except Exception:
            traceback.print_exc()
        return


def test_grid_best(logdirname: str = GRID_LOGDIR) -> None:
    # hparams that NEED to be replicated
    BATCH_SIZE = 16  # NOTE: CURRENTLY ASSUMES BATCH_SIZE IS NOT TUNED
    FEAT_SELECT = "sd"  # no effect with n == N
    NORM: Norm = "feature"  # this is the only one that works

    # options that should be irrelevant, if set correctly (e.g. large enough)
    MAX_STEPS = 3000
    MAX_EPOCHS = 150
    TRAINER_ARGS = dict(max_epochs=MAX_EPOCHS, max_steps=MAX_STEPS)
    REPEATS = 1

    def load_yaml(path: Path) -> Any:
        with open(path, "r") as handle:
            return safe_load(handle)

    # raise NotImplementedError(
    #     "Must still write code to search `logdirname` for `.ckpt` files, "
    #     "extract val_acc from these, and then hparams from .yaml, and then train "
    #     "new versions of each model with random new subjects to show rep fail."
    # )
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
                    model_cls=LinearModel,
                    model_args=model_args,
                    trainer_args=TRAINER_ARGS,
                    logdirname="SimpleLinear--Best",
                )
            except Exception:
                traceback.print_exc()


if __name__ == "__main__":
    # test_exhaustive_grid()
    test_grid_best()
