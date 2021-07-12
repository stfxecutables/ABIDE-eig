import logging
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

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))

from src.analysis.predict.hypertune import HtuneResult, hypertune_classifier
from src.analysis.predict.roi_predict import predict_from_roi_reductions
from src.analysis.rois import identity, max, mean, median, pca, roi_dataframes, std
from src.eigenimage.compute_batch import T_LENGTH
from src.run.cc_setup import RESULTS, setup

LOGFILE = setup("compute_roi_preds")


def compute_results(args: Dict) -> Optional[DataFrame]:
    try:
        scores, guess, htuned = predict_from_roi_reductions(**args)
        params = Namespace(**args)

        print(f"Mean acc: {np.round(np.mean(scores), 3).item()}  (guess = {np.round(guess, 3)})")
        print(
            f"CI: ({np.round(np.percentile(scores, 5), 3)}, {np.round(np.percentile(scores, 95), 3)})"
        )
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
                ),
                **htuned.best_params,
            },
            index=[0],
        ).copy(deep=True)
    except Exception as e:
        msg = f"Got exception {e}"
        logging.debug(f"{msg}\n{traceback.format_exc()}")
        return None


if __name__ == "__main__":
    GRID = dict(
        source=["func", "eigimg"],
        norm=["div", None],
        reducer=[max, mean, std, pca],
        slicer=[slice(None)],
        slice_reducer=[identity],
        weight_sharing=["rois"],
        classifier=[RandomForestClassifier],
        classifier_args=[dict(n_jobs=8)],
    )
    params = list(ParameterGrid(GRID))
    dfs = process_map(compute_results, params, max_workers=5)
    dfs = [df for df in dfs if df is not None]
    df = pd.concat(dfs, axis=0, ignore_index=True)
    df.to_parquet(RESULTS / "roi_results_all.parquet")
