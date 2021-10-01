from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series

ROOT = Path(__file__).resolve().parent
JSONS = sorted(ROOT.rglob("*.json"))


def print_htune_table(df: DataFrame) -> Tuple[DataFrame, str]:
    def renamer(s: str) -> str:
        if "params" not in s:
            s = f"{s}_"
        return s

    def format_time(t: pd.Timedelta) -> str:
        """Convert ms to readable"""
        hrs = t.total_seconds() / 3600
        return f"{hrs:0.1f} hrs"

    shortened = {
        "params_conv_num_layers": "c_n_lay",
        "params_conv_kernel": "c_kern",
        "params_conv_dilation": "c_dil",
        "params_conv_residual": "c_resid",
        "params_conv_depthwise": "c_depth",
        "params_conv_norm": "c_norm",
        "params_conv_norm_groups": "c_n_grp",
        "params_conv_cbam": "cbam",
        "params_conv_cbam_reduction_log2": "cbam_r",
        "params_lstm_num_layers": "l_n_lay",
        "params_lstm_hidden_sizes_log2": "l_n_hid",
        "params_lstm_kernel_sizes": "l_kern",
        "params_lstm_dilations": "l_dil",
        "params_lstm_norm": "l_norm",
        "params_lstm_norm_groups_factor": "l_n_grp",
        "params_lstm_inner_spatial_dropout": "l_drop",
        "params_LR": "LR",
        "params_L2": "L2",
        "value_": "val_acc_max",
        "datetime_start_": "start",
        "datetime_complete_": "complete",
        "duration_": "trained",
        "state_": "state",
        "number_": "id",
    }
    float_fmts = (
        "3.0f",  # number
        "0.3f",  # val_acc
        "3.0f",  # start
        "3.0f",  # complete
        "3.0f",  # trained
        "1.2e",  # L2
        "1.2e",  # LR
        "3.0f",  # cbam
        "3.0f",  # cbamr
        "3.0f",  # c_depth
        "3.0f",  # c_dil
        "3.0f",  # c_kern
        "3.0f",  # c_norm
        "3.0f",  # c_n_grp
        "3.0f",  # c_n_lay
        "3.0f",  # c_resid
        "3.0f",  # l_dil
        "3.0f",  # l_n_hid
        "0.2f",  # l_drop
        "3.0f",  # l_kern
        "3.0f",  # l_norm
        "3.0f",  # l_n_grp
        "3.0f",  # l_n_lay
        "",  # state
    )
    renamed = df.rename(mapper=renamer, axis=1).rename(mapper=shortened, axis=1)
    renamed.start = pd.to_datetime(renamed.start, unit="ms").round("min").astype(str).str[5:-3]
    renamed.complete = (
        pd.to_datetime(renamed.complete, unit="ms").round("min").astype(str).str[5:-3]
    )
    total_time = renamed.trained.sum()
    renamed.trained = renamed.trained.apply(pd.Timedelta, unit="ms").apply(format_time)
    for col in renamed.columns:
        if "system_attrs" in col:
            renamed.drop(columns=col, inplace=True)
    print(renamed.to_markdown(tablefmt="simple", floatfmt=float_fmts, index=False))
    print(f"Total time hypertuning: {format_time(total_time)}")
    return renamed, total_time


if __name__ == "__main__":
    times = []
    for json in JSONS:
        df = pd.read_json(json)
        print(json.name.upper())
        table, time = print_htune_table(df)
        times.append(time.total_seconds() / 3600)
    print(f"Total time tuning all models: {np.sum(times)} hours")
