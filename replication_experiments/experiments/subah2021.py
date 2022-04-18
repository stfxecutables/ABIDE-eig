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

import traceback
from typing import Any, Dict, List

from sklearn.model_selection import ParameterGrid
from torch.nn import Dropout, Linear, ReLU, Sequential

from replication_experiments.constants import Norm
from replication_experiments.evaluation import test_split
from replication_experiments.models import TrainingMixin


class Subah2021(TrainingMixin):
    """https://mdpi-res.com/d_attachment/applsci/applsci-11-03636/article_deploy/applsci-11-03636.pdf

    Subah, F.Z., Deb, K., Dhar, P.K., Koshiba, T. (2021). A Deep Learning Approach to Predict Autism
    Spectrum Disorder Using Multisite Resting-State fMRI. Appl. Sci. 2021, 11, 3636.
    https://doi.org/10.3390/app11083636
    """

    def __init__(
        self,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        guess: float = 0.5,
        in_features: int = 19900,
        dropout: float = 0.8,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(lr=lr, weight_decay=weight_decay, guess=guess, *args, **kwargs)  # type: ignore # noqa
        # self.lr = lr
        # self.weight_decay = weight_decay
        # self.guess = guess

        self.model = Sequential(
            Dropout(dropout),
            Linear(in_features, 32),
            ReLU(inplace=True),
            #
            Dropout(dropout),
            Linear(32, 32),
            ReLU(inplace=True),
            #
            # Dropout(0.8),
            Linear(32, 1),
        )


if __name__ == "__main__":
    # hparams that NEED to be replicated
    N = int((200 ** 2 - 200) / 2)  # upper triangle of 200x200 matrix where diagonals are 1
    BATCH_SIZE = 10
    LR = 1e-4
    FEAT_SELECT = "sd"  # no effect with n == N
    NORM: Norm = "feature"  # this is the only one that works

    # options that should be irrelevant, if set correctly (e.g. large enough)
    MAX_STEPS = 3000
    MAX_EPOCHS = 150
    TRAINER_ARGS = dict(max_epochs=MAX_EPOCHS, max_steps=MAX_STEPS)
    REPEATS = 1

    grid: List[Dict] = list(
        ParameterGrid(
            dict(
                in_features=[N],
                lr=[1e-4],
                weight_decay=[1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1],
                dropout=[0.2, 0.8],
            )
        )
    )

    for i, model_args in enumerate(grid):
        print(f"Iteration {i} of {len(grid)}.")
        print(f"Testing model with args: {model_args}")
        for _ in range(REPEATS):
            try:
                test_split(
                    n_features=N,
                    feat_select=FEAT_SELECT,
                    norm=NORM,
                    batch_size=BATCH_SIZE,
                    model_args=model_args,
                    trainer_args=TRAINER_ARGS,
                    logdirname="subah2021"
                )
            except Exception:
                traceback.print_exc()
            sys.exit()
