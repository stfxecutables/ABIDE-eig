# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.run.cc_setup import setup_environment, setup_logging, RESULTS  # isort:skip
setup_environment()
# fmt: on


import logging
import sys
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from tqdm.contrib.concurrent import process_map

from src.analysis.predict.reducers import eigvals, identity, max, mean, pca, std
from src.analysis.predict.sequence import predict_from_sequence_reductions

LOGFILE = setup_logging("compute_seq_pred")


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
                    acc=htuned.val_acc,
                    guess=guess,
                ),
                **htuned.best_params,
            },
            index=[0],
        ).copy(deep=True)
    except Exception as e:
        msg = f"Got exception {e}"
        logging.debug(f"{msg}\n{traceback.format_exc()}")
        return None


# NOTE: expect this to be about a 4-hour job
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--silent", action="store_true")
    silent = parser.parse_args().silent
    GRID = dict(
        source=["func", "eigimg"],
        norm=["diff", "div", None],  # TODO: Fix this ambiguous behaviour
        reducer=[pca, std, mean, max, eigvals],
        slicer=[slice(None)],
        slice_reducer=[identity],
        classifier=[RandomForestClassifier],
        classifier_args=[dict(n_jobs=-1)],
    )
    params = list(ParameterGrid(GRID))
    dfs = process_map(compute_results, params, disable=silent)
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
