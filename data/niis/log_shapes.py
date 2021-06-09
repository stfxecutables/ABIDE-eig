import pandas as pd
from pandas import DataFrame
from pathlib import Path
from ants import image_read
from tqdm.contrib.concurrent import process_map
from typing import Tuple

NIIS = sorted(Path(__file__).resolve().parent.rglob("*.nii.gz"))

def get_shape(nii: Path):
    return (*image_read(str(nii)).shape, nii)

if __name__ == "__main__":
    rets = process_map(get_shape, NIIS)
    dfs = []
    for ret in rets:
        h, w, d, t, nii = rets
        dfs.append(DataFrame(
            dict(
                H=h,
                W=w,
                D=d,
                T=t
            )
        ), index=[str(nii.name)])
    df = pd.concat(dfs, axis=0)
    print(df)
    outfile = Path(__file__).resolve().parent / "shapes.json"
    df.to_json(outfile)
    print(f"Saved shape data to {outfile}")



