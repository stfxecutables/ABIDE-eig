# fmt: off
from pathlib import Path  # isort:skip
import sys  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

from argparse import ArgumentParser, Namespace
from typing import Any, Dict, Tuple, Type, no_type_check
from warnings import warn

import numpy as np
import torch
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.profiler import AdvancedProfiler

from src.analysis.predict.deep_learning.constants import INPUT_SHAPE

DEFAULTS = Namespace(
    **dict(
        batch_size=4,
        val_batch_size=4,
        num_workers=8,
    )
)
OVERRIDE_DEFAULTS: Dict = dict(
    # https://pytorch-lightning.readthedocs.io/en/stable/extensions/logging.html?highlight=logging#logging-frequency
    log_every_n_steps=5,
    flush_logs_every_n_steps=20,
    max_time={"hours": 2},
)


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
