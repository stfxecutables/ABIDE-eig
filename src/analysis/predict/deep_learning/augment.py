# fmt: off
from pathlib import Path  # isort:skip
import sys  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

from argparse import ArgumentParser, Namespace
from logging import warn
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.profiler import AdvancedProfiler
from scipy.ndimage import gaussian_filter
from torch import Tensor
from torch.nn import (
    AdaptiveMaxPool3d,
    BCEWithLogitsLoss,
    Conv3d,
    InstanceNorm3d,
    Linear,
    ModuleList,
    PReLU,
)
from torch.nn.modules.padding import ConstantPad3d
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, random_split
from torchmetrics.functional import accuracy
from typing_extensions import Literal

from src.analysis.predict.deep_learning.constants import INPUT_SHAPE, PADDED_SHAPE
from src.analysis.predict.deep_learning.dataloader import FmriDataset
from src.analysis.predict.deep_learning.models.layers.conv import ResBlock3d
from src.analysis.predict.deep_learning.models.layers.lstm import ConvLSTM3d
from src.analysis.predict.deep_learning.models.layers.reduce import GlobalAveragePooling
from src.analysis.predict.deep_learning.models.layers.utils import EVEN_PAD
from src.analysis.predict.deep_learning.preprocess import MASK, get_mask_bounds


class Cutout4d:
    """Randomly cut out some amount of the image, up to a max size."""

    SPATIAL_FMRI_MAX = 24

    def __init__(self, max_spatial_size: int, max_temporal_size: int = -1, p: float = 0.5) -> None:
        self.spatial = max_spatial_size
        self.temporal = max_temporal_size
        self.p = np.clip(p, 0, 1)

    def __call__(self, img: ndarray) -> Any:
        # will cutout from start:start + cutsize, so highest start allowed is shape - cutsize
        if np.random.uniform(0, 1) >= self.p:
            return img
        img = np.copy(img)
        maxs = np.array(img.shape)
        starts = np.array([np.random.randint(0, high + 1 - self.spatial) for high in maxs])
        ends = starts + self.spatial
        if self.temporal <= 0:
            img[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2], :] = 0
            return img
        t_start = np.random.randint(0, img.shape[-1] + 1 - self.temporal)
        t_end = t_start + self.temporal
        img[starts[0] : ends[0], starts[1] : ends[1], starts[2] : ends[2], t_start:t_end] = 0
        return img


class Blur4d:
    SIG_SPATIAL_FMRI_MAX = 0.5
    SIG_TEMPORAL_FMRI_MAX = 5.0

    def __init__(
        self, sigma: Union[float, Tuple[float, float, float, float]], p: float = 0.5
    ) -> None:
        self.sigma = sigma
        self.p = np.clip(p, 0, 1)

    def __call__(self, img: ndarray) -> Any:
        if np.random.uniform(0, 1) >= self.p:
            return img
        return gaussian_filter(img, self.sigma, mode="constant", cval=0)


class Remask:
    def __init__(self, fill_value: float = 0) -> None:
        mask = nib.load(str(MASK)).get_fdata().astype(bool)
        cropper = get_mask_bounds(mask)[:-1]
        self.mask = mask[cropper].astype(bool)
        self.fill = fill_value

    def __call__(self, img: ndarray) -> Any:
        img = np.copy(img)
        img[~self.mask] = self.fill
        return img


def _plot_augment(orig: ndarray, aug: ndarray, suptitle: str = "") -> None:
    fig: plt.Figure
    axes: plt.Axes
    ax: plt.Axes
    halves = np.array(orig.shape) // 2
    thirds = np.array(orig.shape) // 3
    others = 2 * thirds
    orig = orig[:, :, :, halves[-1]]  # don't worry about time
    aug = aug[:, :, :, halves[-1]]
    fig, axes = plt.subplots(nrows=2, ncols=9)
    axes[0][0].set_title("Original")
    axes[0][0].imshow(orig[thirds[0], :, :], cmap="Greys")
    axes[0][1].imshow(orig[halves[0], :, :], cmap="Greys")
    axes[0][2].imshow(orig[others[0], :, :], cmap="Greys")
    axes[0][3].imshow(orig[:, thirds[1], :], cmap="Greys")
    axes[0][4].imshow(orig[:, halves[1], :], cmap="Greys")
    axes[0][5].imshow(orig[:, others[1], :], cmap="Greys")
    axes[0][6].imshow(orig[:, :, thirds[2]], cmap="Greys")
    axes[0][7].imshow(orig[:, :, halves[2]], cmap="Greys")
    axes[0][8].imshow(orig[:, :, others[2]], cmap="Greys")

    axes[1][0].set_title("Augmented")
    axes[1][0].imshow(aug[thirds[0], :, :], cmap="Greys")
    axes[1][1].imshow(aug[halves[0], :, :], cmap="Greys")
    axes[1][2].imshow(aug[others[0], :, :], cmap="Greys")
    axes[1][3].imshow(aug[:, thirds[1], :], cmap="Greys")
    axes[1][4].imshow(aug[:, halves[1], :], cmap="Greys")
    axes[1][5].imshow(aug[:, others[1], :], cmap="Greys")
    axes[1][6].imshow(aug[:, :, thirds[2]], cmap="Greys")
    axes[1][7].imshow(aug[:, :, halves[2]], cmap="Greys")
    axes[1][8].imshow(aug[:, :, others[2]], cmap="Greys")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        for spine in ["left", "right", "top", "bottom"]:
            ax.spines[spine].set_visible(False)
    fig.suptitle(suptitle)
    fig.subplots_adjust(top=0.908, bottom=0.03, left=0.035, right=0.988, hspace=0.0, wspace=0.473)
    fig.set_size_inches(h=5, w=14)
    plt.show()


if __name__ == "__main__":
    from src.analysis.predict.deep_learning.dataloader import prepare_data_files

    SIG_SPATIAL = 0.5
    SIGMA = (SIG_SPATIAL, SIG_SPATIAL, SIG_SPATIAL, 5)
    files = prepare_data_files(is_eigimg=False)[:10]
    for file in files:
        orig = np.load(file)
        transforms: List[Callable[[ndarray], ndarray]] = [
            Blur4d(sigma=SIGMA, p=1.0),
            Cutout4d(max_spatial_size=24, p=1.0),
            Remask(),
        ]
        for i, transform in enumerate(transforms):
            aug = transform(orig) if i == 0 else transform(aug)
        _plot_augment(orig, aug, f"sigma={SIGMA}")
