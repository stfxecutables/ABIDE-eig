import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pandas import DataFrame
from tqdm import tqdm

# fmt: off
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
from src.analysis.preprocess.constants import FEATURES_DIR  # isort: skip
# fmt: on


def _show_data_shapes(parent: Path) -> DataFrame:
    files = sorted(parent.rglob("*"))
    exclude = [FEATURES_DIR.name, "cc200", "cc400"]
    parents = np.unique([f.parent for f in files])
    dfs = []
    for p in tqdm(parents):
        if p.name in exclude:
            continue
        label = f"{p.parent.name}/{p.name}"
        if "features_cpac" in label:
            label = label[label.find("/") + 1 :]
        files = p.rglob("*.npy")
        arrs = [np.load(f) for f in files]
        shapes, counts = np.unique([str(a.shape) for a in arrs], return_counts=True)
        for shape, count in zip(shapes, counts):
            dfs.append(DataFrame(dict(feature=label, shape=shape, count=count), index=[0]))
    df = pd.concat(dfs, axis=0, ignore_index=True)
    print(df.to_markdown(tablefmt="simple", index=False))
    pd.options.display.max_rows = 200
    print(df.groupby(["feature", "shape"]).count())
    return df


if __name__ == "__main__":
    _show_data_shapes(FEATURES_DIR)
