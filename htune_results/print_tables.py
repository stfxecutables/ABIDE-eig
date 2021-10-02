from argparse import ArgumentParser
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame, Series

ROOT = Path(__file__).resolve().parent
JSONS = sorted(ROOT.rglob("*.json"))


def min_format(df: DataFrame) -> str:
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
    return df.to_markdown(tablefmt="simple", floatfmt=float_fmts, index=False)


def get_htune_table(df: DataFrame, show: bool = False) -> Tuple[DataFrame, pd.Timedelta]:
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
    renamed = df.rename(mapper=renamer, axis=1).rename(mapper=shortened, axis=1)
    renamed.start = pd.to_datetime(renamed.start, unit="ms").round("min").astype(str).str[5:-3]
    renamed.complete = (
        pd.to_datetime(renamed.complete, unit="ms").round("min").astype(str).str[5:-3]
    )
    renamed.trained = renamed.trained.apply(pd.Timedelta, unit="ms")
    total_time = renamed.trained.sum()
    renamed.trained = renamed.trained.apply(format_time)
    for col in renamed.columns:
        if "system_attrs" in col:
            renamed.drop(columns=col, inplace=True)
    if show:
        print(min_format(renamed))
        print(f"Total time hypertuning: {format_time(total_time)}")
    return renamed, total_time


def get_best_n(df: DataFrame, n: int = 5) -> DataFrame:
    table = df.dropna().sort_values(by="val_acc_max", ascending=False)
    hrs = table.trained.str.replace(" hrs", "").apply(lambda s: float(s))
    table = table.loc[hrs > 0.8]
    best = table.iloc[:n]
    return best


if __name__ == "__main__":
    N = 10
    parser = ArgumentParser()
    parser.add_argument("--long", action="store_true")
    long = parser.parse_args().long
    times = []
    tables: List[DataFrame] = []
    exps = []
    for json in JSONS:
        df = pd.read_json(json)
        experiment = json.stem.upper()
        print(experiment)
        table, time = get_htune_table(df, show=long)
        tables.append(table)
        times.append(time.total_seconds() / 3600)
        exps.append(experiment)
    print("\n\nBest models:\n")
    for table, exp, time in zip(tables, exps, times):
        if exp == "CONV3DTOCONVLSTM3D_EIGIMG":
            exp = "CONV3DTOCONVLSTM3D_EIGIMG_[125,175]"
        best = get_best_n(table, N)
        counts, edges = np.histogram(table.dropna().val_acc_max)
        percents = np.round(100 * counts / np.sum(counts), 2)
        print("=" * 80)
        print(f"{exp} ({np.round(time, 1)} hours / {len(table)} models evaluated)")
        print("=" * 80)
        print("val_acc distribution:\n")
        for i, percent in enumerate(percents):
            print(
                f"  [{np.round(edges[i], 3):0.3f}, {np.round(edges[i+1], 3):0.3f}]: {percent:4.1f}%"
            )
        print(f"\n{exp} Best {N}:")
        print(min_format(best))
    print("=" * 80)
    print(f"Total time tuning all models: {np.sum(times)} hours")
