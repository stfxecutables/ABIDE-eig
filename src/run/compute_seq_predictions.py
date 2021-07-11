import os
import sys
import traceback
from argparse import Namespace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from tqdm.contrib.concurrent import process_map

if os.environ.get("CC_CLUSTER") is not None:
    SCRATCH = os.environ["SCRATCH"]
    os.environ["MPLCONFIGDIR"] = str(Path(SCRATCH) / ".mplconfig")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from src.analysis.predict.sequence import predict_from_sequence_reductions
from src.analysis.rois import identity, max, mean, pca, std

RESULTS = ROOT / "results"
if not RESULTS.exists():
    os.makedirs(RESULTS, exist_ok=True)


def compute_results(args: Dict) -> Optional[DataFrame]:
    try:
        guess, htuned = predict_from_sequence_reductions(**args)
        params = Namespace(**args)
        return DataFrame(
            {
                **dict(
                    model=htuned.classifier,
                    source=params.source,
                    norm=params.norm,
                    roi_reducer=params.reducer.__name__,
                    roi_slicer=str(params.slicer),
                    slice_reducer=params.slice_reducer.__name__,
                    sharing=params.weight_sharing,
                    acc=htuned.val_acc,
                    guess=guess,
                ),
                **htuned.best_params,
            },
            index=[0],
        ).copy(deep=True)
    except Exception as e:
        print(f"Got exception {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    GRID = dict(
        source=["func", "eigimg"],
        norm=["div", None],
        reducer=[max, mean, std, pca],
        slicer=[slice(None)],
        slice_reducer=[identity],
        classifier=[RandomForestClassifier],
        classifier_args=[dict(n_jobs=8)],
    )
    params = list(ParameterGrid(GRID))
    dfs = process_map(compute_results, params, max_workers=len(params))
    dfs = [df for df in dfs if df is not None]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_parquet(RESULTS / "seq_results_all.parquet")

"""
Guess = 0.569
Results: Mean
    Best Acc: 0.591
        source="func", norm="div", reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.556
        source="func", norm="diff", reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.562
        source="func", norm=None, reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,

Results: Std
    Best Acc: 0.551
        source="func", norm="div", reducer=std,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.569
        source="func", norm="diff", reducer=std,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.573
        source="func", norm=None, reducer=std,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,

TODO: test PCA reduced ROIs and/or PCA reduced fMRI images
Results: PCA
    Best Acc: 0.587
        source="func", norm="div", reducer=PCA,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.571
        source="eigimg", norm="diff", reducer=PCA,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.553
        source="eigimg", norm=None, reducer=PCA,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,

Results: Eigimg
    Best Acc: 0.618
        source="func", norm="div", reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.589
        source="func", norm="diff", reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,
    Best Acc: 0.593
        source="func", norm=None, reducer=mean,
        slicer=slice(None), slice_reducer=identity,
        classifier=RandomForestClassifier,

"""
