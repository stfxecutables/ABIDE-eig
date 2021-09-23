from functools import reduce
from pathlib import Path
from typing import no_type_check

import numpy as np
import pandas as pd
import yaml
from pandas import DataFrame
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
from pytorch_lightning.core.saving import load_hparams_from_yaml
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm


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


def load_yaml_unsafe(path: Path) -> DataFrame:
    if not path.exists():
        return None
    res = load_hparams_from_yaml(str(path))
    return res


def get_hparam_info(root: Path) -> DataFrame:
    root = root.resolve()
    event_files = sorted(
        root.rglob("events.out.tfevents.*"), key=lambda p: p.parent.parent.parent.name
    )
    hparams = [load_yaml_unsafe(f.parent / "hparams.yaml") for f in event_files]
    dfs = []
    train_dfs = []
    for event in tqdm(event_files, desc="Converting tfevents"):
        accum = EventAccumulator(str(event), dict(scalars=0))  # load all scalars
        accum.Reload()
        metric_names = [tag for tag in accum.Tags()["scalars"] if tag != "hp_metric"]
        metrics = DataFrame()
        train_metrics = DataFrame()
        for metric in metric_names:
            walltimes, steps, values = zip(*accum.Scalars(metric))
            if "epoch" in metric:
                continue
            if "train" in metric:
                train_metrics[f"wtime"] = walltimes
                train_metrics[f"step"] = steps
                train_metrics[metric] = values
            elif "val" in metric:
                metrics[f"wtime"] = walltimes
                metrics[f"step"] = steps
                metrics[metric] = values
        dfs.append(metrics)
        train_dfs.append(train_metrics)
    return dfs, train_dfs, hparams
