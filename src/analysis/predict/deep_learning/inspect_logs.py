# fmt: off
from pathlib import Path  # isort:skip
import sys  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort:skip
sys.path.append(str(ROOT))  # isort:skip
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy.ndimage import median_filter, uniform_filter1d
from scipy.stats import linregress

from src.analysis.predict.deep_learning.tables import get_hparam_info

if __name__ == "__main__":
    logs = ROOT / "lightning_logs"
    dfs, train_dfs, hparams = get_hparam_info(logs)
    pairs = []
    for df, hp in zip(dfs, hparams):
        if len(df) == 0 or hp is None:
            continue
        end = df.val_acc.to_numpy()[-20:]
        acc = uniform_filter1d(end, 3)
        trend = linregress(range(len(end)), end).slope
        if acc.mean() < 0.63:
            continue
        pairs.append((acc, hp, trend))

    best = sorted(pairs, key=lambda p: p[0].max(), reverse=True)[:20]
    summaries = []
    for accs, hps, trend in best:
        summary = DataFrame()
        summaries.append(
            DataFrame(
                {
                    **{
                        "acc_max": np.max(accs),
                        "acc_mean": np.mean(accs),
                        "trend": trend,
                    },
                    **hps,
                }
            )
        )
        # summary["acc_max"] = np.max(accs)
        # summary["acc_mean"] = np.mean(accs)
        # summary["trend"] = trend
        # for key, val in hps.items():
        #     summary[key] = str(val)
        # summaries.append(summary.copy())
    summary = pd.concat(summaries, axis=0)
    shortened = {
        "conv_num_layers": "c_n_lay",
        "conv_kernel": "c_kern",
        "conv_dilation": "c_dil",
        "conv_residual": "c_resid",
        "conv_depthwise": "c_depth",
        "conv_norm": "c_norm",
        "conv_norm_groups": "c_n_grp",
        "conv_cbam": "cbam",
        "conv_cbam_reduction": "cbam_r",
        "lstm_num_layers": "l_n_lay",
        "lstm_hidden_sizes": "l_n_hid",
        "lstm_in_channels": "l_n_hid",
        "lstm_kernel_sizes": "l_kern",
        "lstm_dilations": "l_dil",
        "lstm_norm": "l_norm",
        "lstm_norm_groups": "l_n_grp",
        "lstm_inner_spatial_dropout": "l_drop",
        "LR": "LR",
        "L2": "L2",
    }
    drop = ["lstm_in_channels", "conv_out_channels", "conv_depthwise_factor"]
    summary = summary.rename(mapper=shortened, axis=1).drop(drop)
    print(summary)
