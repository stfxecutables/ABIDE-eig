# fmt: off
from pathlib import Path  # isort:skip
import sys  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort:skip
sys.path.append(str(ROOT))  # isort:skip
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

import gc
import os
import time
import traceback
import uuid
from argparse import Namespace
from copy import deepcopy
from typing import Any, Dict, Tuple, Type, no_type_check
from warnings import warn

import optuna
import torch
from optuna import Trial
from optuna.pruners import PatientPruner
from pandas import DataFrame
from pytorch_lightning import Trainer, seed_everything
from torch.utils.data import DataLoader

from src.analysis.predict.deep_learning.arguments import get_args
from src.analysis.predict.deep_learning.callbacks import callbacks
from src.analysis.predict.deep_learning.constants import INPUT_SHAPE
from src.analysis.predict.deep_learning.dataloader import FmriDataset
from src.analysis.predict.deep_learning.models.conv_lstm import Conv3dToConvLstm3d
from src.analysis.predict.deep_learning.tables import tableify_logs

"""
Notes
-----
As per http://proceedings.mlr.press/v28/bergstra13.pdf, we should not really
expect Tree-Parzen Estimators to beat random search until about 200 trials or
so. This is not a reasonable number of trials in our budget, *unless* we do
very agressive early-stopping. So we have to get this right. Unfortunately,
Optuna handles this poorly
"""

HTUNE_RESULTS = ROOT / "htune_results"
os.makedirs(HTUNE_RESULTS, exist_ok=True)


def print_htune_table(df: DataFrame) -> None:
    def renamer(s: str) -> str:
        if "params" not in s:
            s = f"{s}_"
        return s

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
    }
    renamed = df.rename(mapper=renamer, axis=1).rename(mapper=shortened, axis=1)
    print(renamed.to_markdown(tablefmt="simple", floatfmt="0.3f"))


def conv3d_to_lstm32_config(args: Namespace, trial: Trial) -> Namespace:
    T = INPUT_SHAPE[0]
    idx = list(range(T))
    if args.slicer is None:
        raise RuntimeError("Optuna config can be build only on updated `args` object.")
    channels = len(idx[args.slicer])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 2)
    hidden_sizes = 2 ** trial.suggest_int("lstm_hidden_sizes_log2", 2, 6)
    kernel_sizes = trial.suggest_int("lstm_kernel_sizes", 3, 5, 2)
    dilations = trial.suggest_int("lstm_dilations", 1, 3)
    lstm_hidden_sizes = [hidden_sizes for _ in range(lstm_num_layers)]
    lstm_kernel_sizes = [kernel_sizes for _ in range(lstm_num_layers)]
    lstm_dilations = [dilations for _ in range(lstm_num_layers)]
    # fmt: off
    return Namespace(
        **dict(
            conv_in_channels=channels,
            conv_out_channels=channels,
            conv_num_layers=trial.suggest_int("conv_num_layers", 2, 4),
            conv_kernel_size=trial.suggest_int("conv_kernel", 3, 7, 2),
            conv_dilation=trial.suggest_int("conv_dilation", 1, 3),
            conv_residual=trial.suggest_categorical("conv_residual", [True, False]),
            conv_halve=True,
            conv_depthwise=trial.suggest_categorical("conv_depthwise", [True, False]),
            conv_depthwise_factor=None,
            conv_norm=trial.suggest_categorical("conv_norm", ["group", "batch"]),
            conv_norm_groups=5**trial.suggest_int("conv_norm_groups", 0, 2),
            conv_cbam=trial.suggest_categorical("conv_cbam", [True, False]),
            conv_cbam_reduction=2**trial.suggest_int("conv_cbam_reduction_log2", 1, 4),
            lstm_in_channels=1,
            lstm_num_layers=lstm_num_layers,
            lstm_hidden_sizes=lstm_hidden_sizes,
            lstm_kernel_sizes=lstm_kernel_sizes,
            lstm_dilations=lstm_dilations,
            lstm_norm=trial.suggest_categorical("lstm_norm", ["group", "batch"]),
            lstm_norm_groups=trial.suggest_int("lstm_norm_groups_factor", 2, 4, 2),  # must divide hidden_sizes # noqa
            lstm_inner_spatial_dropout=trial.suggest_float("lstm_inner_spatial_dropout", 0, 0.6),
            lr=trial.suggest_loguniform("LR", 1e-6, 1e-3),
            l2=trial.suggest_loguniform("L2", 1e-7, 1e-3),
            trial_id=trial.number,
            uuid=str(uuid.uuid4()),
        )
    )
    # fmt: on


def suggest_config(args: Namespace, model: Type, trial: Trial) -> Namespace:
    func_map = {Conv3dToConvLstm3d.__name__: conv3d_to_lstm32_config}
    return func_map[model.__name__](args, trial)


class Objective:
    def __init__(self, model_class: Type, args: Namespace) -> None:
        self.model_class = model_class
        self.args = args

    def __call__(self, trial: Trial) -> float:
        seed_everything(333)
        config = suggest_config(self.args, self.model_class, trial)
        args = deepcopy(self.args)
        args.default_root_dir = self.args.default_root_dir / config.uuid
        model = self.model_class(config)
        trainer = Trainer.from_argparse_args(args, callbacks=callbacks(trial))
        trainer.logger.log_hyperparams(config)
        train, val = FmriDataset(args).train_val_split(args)
        train_loader = DataLoader(
            train,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            val,
            batch_size=args.val_batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            drop_last=False,
        )
        try:
            trainer.fit(model, train_loader, val_loader)
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Ran out of memory. Cleaning up and returning zero acc.")
                train_loader = val_loader = self.train = self.val = trainer = model = None  # type: ignore # noqa
                gc.collect()
                time.sleep(10)  # Optuna too dumb to wait for CPU memory to free
                return 0.0
            else:
                raise e
        try:
            torch.cuda.empty_cache()
        except Exception:
            print("Got an error clearing CUDA cache")
            traceback.print_exc()
        try:
            df = tableify_logs(trainer)
        except:  # noqa
            traceback.print_exc()
            warn("No data logged to dataframes, returning 0. Traceback above.")
            train_loader = val_loader = self.train = self.val = trainer = model = None  # type: ignore # noqa
            gc.collect()
            time.sleep(10)  # Optuna too dumb to wait for CPU memory to free
            return 0.0

        train_loader = val_loader = self.train = self.val = trainer = model = None  # type: ignore
        gc.collect()
        time.sleep(10)  # Optuna too dumb to wait for CPU memory to free
        return float(df["val_acc"].max())


if __name__ == "__main__":
    almost_day = int(23.5 * 60 * 60)
    model_class = Conv3dToConvLstm3d
    args = get_args(model_class)
    objective = Objective(model_class, args)

    # optuna.logging.get_logger("optuna").addHandler(StreamHandler(sys.stdout))
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study_name = f"{Conv3dToConvLstm3d.__name__}_{'eigimg' if args.is_eigimg else 'fmri'}"
    storage_name = f"sqlite:///{HTUNE_RESULTS}/{study_name}.db"
    study = optuna.create_study(
        storage=storage_name,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        # note patience also effectively sets a min number of steps before pruning
        pruner=PatientPruner(optuna.pruners.SuccessiveHalvingPruner(), patience=30),
        study_name=study_name,
        direction="maximize",
        load_if_exists=True,
    )
    print("Resuming from previous study with data: ")
    print_htune_table(study.trials_dataframe())
    study.optimize(
        objective,
        n_trials=400,
        timeout=almost_day,
        gc_after_trial=True,
        show_progress_bar=False,
    )
    # df = study.trials_dataframe()
    # df.to_json(HTUNE_RESULTS / f"{study_name}_trials.json")
    print(f"Updated hypertuning results for {model_class.__name__}:")
    print_htune_table(df)
