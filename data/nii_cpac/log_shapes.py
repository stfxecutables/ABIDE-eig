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


def get_info(nii: Path) -> DataFrame:
    img = image_read(str(nii))
    shape = str(img.shape[:-1]).replace("(", "").replace(")", "").replace(", ", ",")
    t = img.shape[-1]
    dt = img.spacing[-1]
    df = DataFrame(dict(shape=shape, T=t, dt=dt), index=[str(nii.stem)])
    return df


if __name__ == "__main__":
    dfs = process_map(get_info, NIIS)
    df = pd.concat(dfs, axis=0)
    print(df)
    df.to_json(OUTFILE)
    print(f"Saved shape data to {OUTFILE}")
