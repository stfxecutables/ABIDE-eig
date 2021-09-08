# fmt: off
from pathlib import Path  # isort:skip
import sys  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort:skip
sys.path.append(str(ROOT))  # isort:skip
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

import uuid
from argparse import Namespace
from typing import Any, Dict, Tuple, Type, no_type_check
from warnings import warn

import optuna
from optuna import Trial
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
            conv_norm_groups=5**trial.suggest_int("conv_norm_groups", 0, 1, 2),
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
            uuid=str(uuid.uuid4()),
        )
    )
    # fmt: on


def suggest_config(args: Namespace, model: Type, trial: Trial) -> Namespace:
    func_map = {Conv3dToConvLstm3d.__name__: conv3d_to_lstm32_config}
    return func_map[model.__name__](args, trial)


def train_model(
    model_class: Type,
) -> None:
    seed_everything(333)
    args = get_args(model_class)
    config = model_class.config(args)

    data = FmriDataset(args)
    train, val = data.train_val_split(args)
    model = model_class(config)
    trainer = Trainer.from_argparse_args(args, callbacks=callbacks(config))
    trainer.logger.log_hyperparams(config)
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
    trainer.fit(model, train_loader, val_loader)
    tableify_logs(trainer)


class Objective:
    def __init__(self, model_class: Type, args: Namespace, data: FmriDataset) -> None:
        self.model_class = model_class
        self.args = get_args(self.model_class)
        self.train, self.val = data.train_val_split(args)
        pass

    def __call__(self, trial: Trial) -> float:
        seed_everything(333)
        config = suggest_config(self.args, self.model_class, trial)
        model = self.model_class(config)
        trainer = Trainer.from_argparse_args(self.args, callbacks=callbacks(config, trial))
        trainer.logger.log_hyperparams(config)
        train_loader = DataLoader(
            self.train,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            shuffle=True,
            drop_last=False,
        )
        val_loader = DataLoader(
            self.val,
            batch_size=self.args.val_batch_size,
            num_workers=self.args.num_workers,
            shuffle=False,
            drop_last=False,
        )
        trainer.fit(model, train_loader, val_loader)
        try:
            df = tableify_logs(trainer)
            return float(df["val_acc"].max())
        except:
            warn("No data logged to dataframes, returning 0")
            return 0.0


if __name__ == "__main__":
    almost_day = int(23.5 * 60 * 60)
    model_class = Conv3dToConvLstm3d
    args = get_args(model_class)
    data = FmriDataset(args)
    objective = Objective(model_class, args, data)

    # optuna.logging.get_logger("optuna").addHandler(StreamHandler(sys.stdout))
    optuna.logging.set_verbosity(optuna.logging.DEBUG)
    study_name = f"{Conv3dToConvLstm3d.__name__}_{'eigimg' if args.is_eigimg else 'fmri'}"
    storage_name = f"sqlite:///{study_name}.db"
    study = optuna.create_study(
        storage=storage_name,
        sampler=optuna.samplers.TPESampler(n_startup_trials=10),
        pruner=optuna.pruners.SuccessiveHalvingPruner(min_resource=10),
        study_name=study_name,
        direction="maximize",
        load_if_exists=True,
    )
    print("Resuming from previous study with data: ")
    print(study.trials_dataframe().to_markdown(tablefmt="simple", floatfmt="0.2f"))
    study.optimize(objective, n_trials=200, timeout=almost_day)
    df = study.trials_dataframe()
    df.to_json("trials_df.json")
    print(df.to_markdown(tablefmt="simple", floatfmt="0.3f"))
