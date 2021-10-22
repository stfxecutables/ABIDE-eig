import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import sys
from argparse import ArgumentParser
from pathlib import Path
from typing import List

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants.paths import NIIS
from src.eigenimage.compute import compute_eigenimage

SHAPES = NIIS / "shapes.json"

# we'll be cutting the timepoints to the last 176, which means computations
# take very close to 2 hours each time, so 11 computations per 24h window,
# i.e. 11 computations per batch
MAX_JOB_TIME = 12  # hours
EIGIMG_COMPUTE_TIME = 1 / 3  # new masked images take about 15 minutes for 176 timepoints
T_LENGTH = 176

# computed / derived based on constants above
BATCH_SIZE = int(MAX_JOB_TIME / EIGIMG_COMPUTE_TIME) - 1  # -1 to be careful / give extra time


def get_batch_idx() -> int:
    parser = ArgumentParser()
    parser.add_argument("--batch", type=int)
    args = parser.parse_args()
    return int(args.batch)


def get_files() -> List[Path]:
    df = pd.read_json(SHAPES)
    tr_2 = df.loc[df.dt == 2.0, :].sort_index()
    files = [NIIS / file for file in tr_2.index.to_list()]
    return files


if __name__ == "__main__":
    files = get_files()
    idx = get_batch_idx()
    batch = files[idx * BATCH_SIZE : (idx + 1) * BATCH_SIZE]
    for file in batch:
        compute_eigenimage(file, covariance=True, t=T_LENGTH)
