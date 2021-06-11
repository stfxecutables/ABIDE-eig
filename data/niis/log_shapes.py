import os
from pathlib import Path
from typing import Tuple

import pandas as pd
from ants import image_read
from pandas import DataFrame
from tqdm.contrib.concurrent import process_map

NIIS = sorted(Path(__file__).resolve().parent.rglob("*.nii.gz"))
CC_CLUSTER = os.environ.get("CC_CLUSTER")
SCRATCH = Path(__file__).resolve().parent if CC_CLUSTER is None else Path(os.environ.get("SCRATCH"))
OUTFILE = SCRATCH / "shapes.json"


def get_shape(nii: Path):
    img = image_read(str(nii))
    shape = img.shape
    dt = img.spacing[-1]
    return shape, dt, nii


if __name__ == "__main__":
    rets = process_map(get_shape, NIIS)
    dfs = []
    for ret in rets:
        shape, dt, nii = ret
        h, w, d, t = shape
        dfs.append(DataFrame(dict(H=h, W=w, D=d, T=t, dt=dt), index=[str(nii.name)]))
    df = pd.concat(dfs, axis=0)
    print(df)
    df.to_json(OUTFILE)
    print(f"Saved shape data to {OUTFILE}")
