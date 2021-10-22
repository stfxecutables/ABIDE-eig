from __future__ import annotations  # isort:skip # noqa

import os
import sys
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple, Union, cast
from warnings import warn

import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray
from typing_extensions import Literal

# fmt: off
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

# HACK for local testing
if os.environ.get("CC_CLUSTER") is None:
    os.environ["CC_CLUSTER"] = "home"

from data.download_cpac_1035 import download_csv

from src.analysis.features import FEATURES, Feature
from src.analysis.preprocess.atlas import Atlas
from src.analysis.preprocess.constants import ATLASES, FEATURES_DIR, T_CROP

f64Array = NDArray[f64]
Arrays = List[f64Array]

"""Define splitting procedures for reproducible and sane validation of ABIDE data

Notes
-----
Must implement:

    - stratified splitting by site
    - leave-one-site-out
    - repeatable (robust) k-fold

"""

if __name__ == "__main__":
    pass
