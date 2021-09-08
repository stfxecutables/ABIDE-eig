from functools import reduce
from pathlib import Path
from typing import no_type_check

import numpy as np
import pandas as pd
from pandas import DataFrame
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def merge_dfs(df1: DataFrame, df2: DataFrame) -> DataFrame:
    left = df1  # only keep first time column
    right = df2.drop(columns="wtime")
    return pd.merge(left, right, how="inner", on="step")


def tableify_logs(trainer: Trainer) -> DataFrame:
    # magic below from https://stackoverflow.com/a/45899735
    logdir = trainer.logger.experiment.log_dir
    accum = EventAccumulator(logdir)
    accum.Reload()
    metric_names = [tag for tag in accum.Tags()["scalars"] if tag != "hp_metric"]
    train_metrics, val_metrics, other_metrics = {}, {}, {}
    for metric in metric_names:
        walltimes, steps, values = zip(*accum.Scalars(metric))
        if "train" in metric:
            train_metrics[metric] = DataFrame({"wtime": walltimes, "step": steps, metric: values})
        elif "val" in metric:
            val_metrics[metric] = DataFrame({"wtime": walltimes, "step": steps, metric: values})
        else:
            other_metrics[metric] = DataFrame({"wtime": walltimes, "step": steps, metric: values})

    df = reduce(merge_dfs, val_metrics.values())
    pearsons_all = df.corr()
    kendalls_all = df.corr(method="kendall")
    acc_pearsons = pearsons_all.loc[:, "val_acc"]
    acc_kendalls = kendalls_all.loc[:, "val_acc"]

    outdir = Path(logdir).resolve()
    df.to_json(outdir / "metrics.json")
    pearsons_all.to_json(outdir / "pearson.json")
    kendalls_all.to_json(outdir / "kendall.json")

    print(df.to_markdown(tablefmt="simple", floatfmt="1.3f"))
    print("\nPearson correlations:")
    print(pearsons_all.to_markdown(tablefmt="simple", floatfmt="1.2f"))
    print(acc_pearsons.to_markdown(tablefmt="simple", floatfmt="1.2f"))
    print("\nKendall correlations:")
    print(kendalls_all.to_markdown(tablefmt="simple", floatfmt="1.2f"))
    print(acc_kendalls.to_markdown(tablefmt="simple", floatfmt="1.2f"))
    return df


def save_test_results(trainer: Trainer) -> None:
    res = trainer.test()[0]
    logdir = trainer.logger.experiment.log_dir
    outdir = Path(logdir).resolve()
    DataFrame(res, index=[trainer.model.uuid]).to_json(outdir / "test_acc.json")


@no_type_check
def save_predictions(
    model: LightningModule, datamodule: LightningDataModule, trainer: Trainer
) -> None:
    logdir = trainer.logger.experiment.log_dir
    outdir = Path(logdir).resolve()
    preds, y_true, niis = zip(*trainer.predict(model, datamodule=datamodule))
    preds = np.array(preds)
    y_true = np.array(y_true)
    niis = [nii[0] for nii in niis]
    preds_df = DataFrame(dict(y_pred=preds, y_true=y_true, nii=niis))
    print(preds_df.to_markdown(tablefmt="simple"))
    preds_df.to_json(outdir / "predictions.json")
