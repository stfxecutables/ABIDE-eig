# fmt: off
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from typing_extensions import Literal

ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)
logging.getLogger("tensorflow").setLevel(logging.FATAL)
logging.getLogger("tensorboard").setLevel(logging.FATAL)
# fmt: on

LOGS = ROOT / "lightning_logs"

Norm = Optional[Literal["const", "feature", "grand"]]
