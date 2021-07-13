# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.run.cc_setup import setup_environment, setup_logging  # isort:skip
setup_environment()
# fmt: on


import logging
import sys
import traceback
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from pandas.core.frame import DataFrame
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import ParameterGrid
from tqdm.contrib.concurrent import process_map

from src.analysis.predict.roi_predict import predict_from_roi_reductions
from src.analysis.rois import identity, max, mean, pca, std
from src.run.cc_setup import RESULTS

LOGFILE = setup_logging("compute_roi_preds")


def compute_results(args: Dict) -> Optional[DataFrame]:
    try:
        scores, guess, htuned = predict_from_roi_reductions(**args)
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
        msg = f"Got exception {e}"
        logging.debug(f"{msg}\n{traceback.format_exc()}")
        return None


# NOTE: For the GRID
#
#    GRID = dict(
#        source=["func", "eigimg"],
#        norm=["div", None],
#        reducer=[max, mean, std, pca],
#        slicer=[slice(None)],
#        slice_reducer=[identity],
#        weight_sharing=["rois"],
#        classifier=[RandomForestClassifier],
#        classifier_args=[dict(n_jobs=-1)],
#    )
#
# Runtime n_trials=10 is about 8 minutes
# For n_trials=200, actual runtime was 2:31:03 (~2.5hrs)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--silent", action="store_true")
    silent = parser.parse_args().silent
    GRID = dict(
        source=["func", "eigimg"],
        norm=["div", None],
        reducer=[max, mean, std, pca],
        slicer=[slice(None)],
        slice_reducer=[identity],
        weight_sharing=["rois"],
        classifier=[RandomForestClassifier],
        classifier_args=[dict(n_jobs=-1)],
    )
    params = list(ParameterGrid(GRID))
    dfs = process_map(compute_results, params, disable=silent)
    dfs = [df for df in dfs if df is not None]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_parquet(RESULTS / "roi_results_all.parquet")
