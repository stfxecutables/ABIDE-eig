# fmt: off
from pathlib import Path  # isort:skip
import sys  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on
import os
import uuid
from argparse import ArgumentParser, Namespace
from typing import Dict, Type

from pytorch_lightning import Trainer
from pytorch_lightning.profiler import AdvancedProfiler

DEFAULTS = Namespace(
    **dict(
        batch_size=4,
        val_batch_size=4,
        num_workers=8,
    )
)
OVERRIDE_DEFAULTS: Dict = (
    dict(
        # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html?highlight=logging#logging-frequency
        log_every_n_steps=5,
        # flush_logs_every_n_steps=20,
        max_time={"hours": 2},
    )
    if os.environ.get("CC_CLUSTER") is None
    else dict(max_time={"hours": 2})
)
PBAR = 0 if os.environ.get("CC_CLUSTER") is not None else 1


def update_args(args: Namespace, model_class: Type) -> Namespace:
    is_eigimg = args.is_eigimg
    root_dir = ROOT / f"lightning_logs/{model_class.__name__}/{'eigimg' if is_eigimg else 'func'}"
    args.default_root_dir = root_dir
    for key, value in OVERRIDE_DEFAULTS.items():
        setattr(args, key, value)
    if args.profile:
        profiler = AdvancedProfiler(dirpath=None, filename="profiling", line_count_restriction=2.0)
        args.profiler = profiler
    args.slicer = slice(args.slice_start, args.slice_end)
    # Compute Canada overrides
    args.progress_bar_refresh_rate = PBAR
    return args

def update_test_args(args: Namespace, model_class: Type) -> Namespace:
    is_eigimg = args.is_eigimg
    root_dir = ROOT / f"lightning_logs_test/{model_class.__name__}/{'eigimg' if is_eigimg else 'func'}"
    args.default_root_dir = root_dir
    if args.profile:
        profiler = AdvancedProfiler(dirpath=None, filename="profiling", line_count_restriction=2.0)
        args.profiler = profiler
    args.slicer = slice(args.slice_start, args.slice_end)
    # Compute Canada overrides
    args.progress_bar_refresh_rate = PBAR
    return args


def get_args(model_class: Type) -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=DEFAULTS.batch_size, type=int)
    parser.add_argument("--val_batch_size", default=DEFAULTS.val_batch_size, type=int)
    parser.add_argument("--num_workers", default=DEFAULTS.num_workers, type=int)
    parser.add_argument("--is_eigimg", action="store_true")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--slice_start", type=int, default=None)
    parser.add_argument("--slice_end", type=int, default=None, help="index one past last timepoint")
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args = update_args(args, model_class)
    return args


def get_conv3d_to_lstm3d_config() -> Namespace:
    from src.analysis.predict.deep_learning.models.conv_lstm import Conv3dToConvLstm3d

    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=DEFAULTS.batch_size, type=int)
    parser.add_argument("--val_batch_size", default=DEFAULTS.val_batch_size, type=int)
    parser.add_argument("--num_workers", default=DEFAULTS.num_workers, type=int)
    parser.add_argument("--is_eigimg", action="store_true")
    parser.add_argument("--preload", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--slice_start", type=int, default=None)
    parser.add_argument("--slice_end", type=int, default=None, help="index one past last timepoint")

    parser.add_argument("--conv_in_channels", type=int)
    parser.add_argument("--conv_out_channels", type=int)
    parser.add_argument("--conv_num_layers", type=int)
    parser.add_argument("--conv_kernel_size", type=int)
    parser.add_argument("--conv_dilation", type=int)
    parser.add_argument("--conv_residual", action="store_true")
    parser.add_argument("--conv_halve", action="store_true", default=True)
    parser.add_argument("--conv_depthwise", action="store_true")
    parser.add_argument("--conv_depthwise_factor", type=int)
    parser.add_argument("--conv_norm", choices=["group", "batch"])
    parser.add_argument("--conv_norm_groups", type=int)
    parser.add_argument("--conv_cbam", action="store_true")
    parser.add_argument("--conv_cbam_reduction", type=int, choices=[2, 4, 8, 16])
    parser.add_argument("--lstm_in_channels", type=int, default=1)
    parser.add_argument("--lstm_num_layers", type=int)
    parser.add_argument("--lstm_hidden_sizes", nargs="+", type=int)
    parser.add_argument("--lstm_kernel_sizes", nargs="+", type=int)
    parser.add_argument("--lstm_dilations", nargs="+", type=int)
    parser.add_argument("--lstm_norm", choices=["group", "batch"])
    parser.add_argument("--lstm_norm_groups", type=int, choices=[2, 4])
    parser.add_argument("--lstm_inner_spatial_dropout", type=float)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--l2", type=float)
    parser.add_argument("--uuid", type=str, default=str(uuid.uuid4()))
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args = update_test_args(args, Conv3dToConvLstm3d)
    return args
