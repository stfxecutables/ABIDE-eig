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

from typing import Any

from torch.nn import Dropout, Linear, ReLU, Sequential

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
