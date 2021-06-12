import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.constants import NII_PATH

NIIS = sorted(NII_PATH.glob("*minimal.nii.gz"))
INFO = pd.read_json(NII_PATH).drop(columns=["H", "W", "D"])
