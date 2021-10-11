from __future__ import annotations  # isort:skip # noqa

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from numpy import ndarray
from pandas import DataFrame, Series
from tqdm import tqdm
from typing_extensions import Literal

# fmt: off
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from data.download_cpac_1035 import download_csv

from src.analysis.preprocess.atlas import Atlas
from src.analysis.preprocess.constants import ATLASES, FEATURES_DIR, T_CROP

SUBJ_DATA = download_csv().loc[:, ["fname", "DX_GROUP"]]
# fix idiotic default of 1 == ASD, 2 == TD, so that now
# 1 = autism, 0 = TD (typical development, control)
SUBJ_DATA.DX_GROUP = SUBJ_DATA.DX_GROUP.apply(lambda x: 2 - x)
SUBJ_DATA.index = pd.Index(SUBJ_DATA.fname.values, name="fname")
SUBJ_DATA.drop(columns="fname", inplace=True)


def get_class(file: Path) -> int:
    stem = file.stem
    fname = stem[: stem.find("__")]
    return int(SUBJ_DATA.loc[fname])


@dataclass
class Shape:
    shape: Tuple[int, ...]
    mins: Tuple[int, ...]
    maxs: Tuple[int, ...]

    def __str__(self) -> str:
        ranges = []
        for mn, mx in zip(self.mins, self.maxs):
            ranges.append(f"[{mn},{mx}]")
        rng = f"{' x '.join(ranges)}"
        info = f"{str(self.shape):^12} in {rng:>25}"
        return f"Shape: {info}"

    __repr__ = __str__


class Feature:
    """Class to contain everything needed to load data and fit a model (hopefully). """

    def __init__(self, name: str, atlas: Optional[Atlas] = None) -> None:
        self.name: str = name
        self.atlas: Optional[Atlas] = atlas
        self.path: Path = self.get_path(self.name, self.atlas)
        self.shape_data: Shape = self.get_shape(self.name, self.atlas)

    def load(self, normalize: bool = True, stack: bool = True) -> Tuple[ndarray, ndarray]:
        """Returns data in form of (x, y), where `y` is the labels (0=TD, 1=ASD)

        Parameters
        ----------
        normalize: bool = True
            Return the feature normalized via its particular normalization strategy.

        stack: bool = True
            Combine the features in a single numpy array where the first dimension is the subject
            (batch) dimension.  If the feature has variable-length time dimension (first dimension),
            it will be cropped and/or padded with a method appropriate to the feature.
        """
        files = sorted(self.path.rglob("*.npy"))
        arrs = [np.load(f) for f in files]
        y = np.array([get_class(f) for f in files])
        if normalize:  # NOTE: MUST normalize first, before padding or whatever
            pass
        if not stack:
            return arrs, y
        x = self._stack_subjects(arrs)
        return x, y

    def _stack_subjects(self, arrs: List[ndarray], unified_size: int = 200) -> ndarray:
        if self.shape_data.shape[0] != -1:  # just works
            return np.stack(arrs, axis=0)
        if self.name == "eig_full_pc":  # unification was already handled
            return np.stack(arrs, axis=0)
        # now only two cases to handle, roi_means/roi_sds, and eig_full_/c/p
        if "roi" in self.name:
            # front-pad short sequences to 200 with zeroes, crop long ones
            unified, T = [], unified_size
            for arr in arrs:
                t = arr.shape[0]
                if t > T:
                    unified.append(arr[:T, :])
                elif t < T:
                    unified.append(np.pad(arr, ((T - t, 0), (0, 0))))
                else:
                    unified.append(arr)
            return np.stack(unified, axis=0)
        if "eig_full" in self.name:
            # this case is unpleasant... front zero pad to longest for now
            unified, T = [], max([arr.shape[0] for arr in arrs])
            for arr in arrs:
                t = arr.shape[0]
                if t > T:
                    unified.append(arr[:T, :])
                elif t < T:
                    unified.append(np.pad(arr, (T - t, 0)))
                else:
                    unified.append(arr)
            return np.stack(unified, axis=0)

        return np.stack(arrs, axis=0)  # simple cases

    def inspect(self) -> None:
        """Plot distributions, report stats, etc.

        Notes
        -----
        For 1D data, plot histogram and stats.
        For 2D data:
            - if ROI summary signals, plot waves
            - if matrix, plot heatmap
        """
        shape = self.shape_data.shape
        if len(shape) == 1:
            x, y = self.load(normalize=False, stack=True)
            x_asd, x_td = x[y == 0], x[y == 1]
            fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
            axes[1][1].sharex(axes[1][0])
            axes[1][2].sharex(axes[1][0])
            axes[1][1].sharey(axes[1][0])
            axes[1][2].sharey(axes[1][0])

            axes[0][1].sharey(axes[0][0])
            axes[0][2].sharey(axes[0][0])
            axes[0][1].sharex(axes[0][0])
            axes[0][2].sharex(axes[0][0])

            info_all = self.plot_hist(axes[1][0], x, "All subjects")
            info_asd = self.plot_hist(axes[1][1], x_asd, "ASD")
            info_td = self.plot_hist(axes[1][2], x_td, "TD")
            # self.plot_curves(axes[0][0], x, "All subjects")
            self.plot_curves(axes[0][0], x_asd, "ASD", color="#ffa514")
            self.plot_curves(axes[0][0], x_td, "TD", color="#146eff")
            axes[0][0].set_title("All Subjects")
            self.plot_curves(axes[0][1], x, "ASD")
            self.plot_curves(axes[0][2], x, "TD")
            fig.text(x=0.16, y=0.05, s=info_all)
            fig.text(x=0.45, y=0.05, s=info_asd)
            fig.text(x=0.72, y=0.05, s=info_td)
            fig.subplots_adjust(
                top=0.91, bottom=0.22, left=0.125, right=0.9, hspace=0.32, wspace=0.2
            )
            fig.set_size_inches(h=8, w=16)
            fig.suptitle(self.name)
            plt.show()
            return
        # now either we have a matrix, or actual waveforms

    @staticmethod
    def plot_hist(ax: plt.Axes, x: ndarray, label: str) -> str:
        # overall stats
        mn, p5, p10, med, p90, p95, mx = np.nanpercentile(
            x, [0, 5, 10, 50, 90, 95, 100], axis=(0, 1)
        )
        # subject variance
        sd_mn, sd_p5, sd_p10, sd_med, sd_p90, sd_p95, sd_mx = np.std(
            np.nanpercentile(x, [0, 5, 10, 50, 90, 95, 100], axis=1), axis=1
        )
        sd_info = (
            f"Btw. subject sds:\n"
            f"  min/max: [{sd_mn:1.1e}, {sd_mx:1.1e}]\n"
            f"   5%/95%: [{sd_p5:1.1e}, {sd_p95:1.1e}]\n"
            f"  10%/90%: [{sd_p10:1.1e}, {sd_p90:1.1e}]\n"
            f"   median:  {sd_med:1.1e}"
        )
        counts = ax.hist((x[~np.isnan(x)]).ravel(), bins=200, color="black")[0]
        ymax = np.max(counts)
        ymax += 0.1 * ymax
        ax.vlines(mn, color="#f40101", label="min all", ymin=0, ymax=ymax, lw=0.5, alpha=0.5)
        ax.vlines(p5, color="#f47201", label="5% all", ymin=0, ymax=ymax, lw=0.5, alpha=0.5)
        ax.vlines(p10, color="#f4e701", label="10% all", ymin=0, ymax=ymax, lw=0.5, alpha=0.5)
        ax.vlines(med, color="#22f401", label="med all", ymin=0, ymax=ymax, lw=0.5, alpha=0.5)
        ax.vlines(p90, color="#017af4", label="90% all", ymin=0, ymax=ymax, lw=0.5, alpha=0.5)
        ax.vlines(p95, color="#4a01f4", label="95% all", ymin=0, ymax=ymax, lw=0.5, alpha=0.5)
        ax.vlines(mx, color="#a701f4", label="max all", ymin=0, ymax=ymax, lw=0.5, alpha=0.5)

        ax.set_xlabel("Feature values")
        ax.set_ylabel("Count")
        # ax.set_xscale("log")
        ax.set_yscale("log")
        ax.legend()
        ax.set_title(label)
        return sd_info

    def plot_curves(self, ax: plt.Axes, x: ndarray, label: str, color: str = "black") -> None:
        alpha = 0.2 if color == "black" else 0.3
        for i in range(x.shape[0]):
            lab = label if (i == 0) else None
            ax.plot(x[i], color=color, alpha=alpha, lw=0.75, label=lab)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Feature value")
        # ax.set_xscale("log")
        if "eig_" in self.name:
            ax.set_yscale("log")
        ax.set_title(label)
        ax.legend()

    @staticmethod
    def get_path(name: str, atlas: Optional[Atlas] = None) -> Path:
        if atlas is None:
            return Path(FEATURES_DIR / name)
        return Path(FEATURES_DIR / f"{atlas.name}/{name}")

    @staticmethod
    def get_shape(name: str, atlas: Optional[Atlas] = None) -> Shape:
        TMAX = 316
        TMIN = 78
        # fmt: off
        if atlas is None:
            return {
                "eig_full":    Shape((-1,),         mins=(TMIN - 1,),   maxs=(295,)),
                "eig_full_c":  Shape((-1,),         mins=(TMIN - 1,),   maxs=(T_CROP - 1,)),
                "eig_full_p":  Shape((-1,),         mins=(T_CROP - 1,), maxs=(TMAX - 1,)),
                "eig_full_pc": Shape((T_CROP - 1,), mins=(T_CROP - 1,), maxs=(T_CROP - 1,)),
            }[name]
        # fmt: on

        roi = 200 if atlas.name == "cc200" else 392
        # fmt: off
        return {
            "eig_mean":   Shape((roi,),     mins=(roi,),      maxs=(roi,)),
            "eig_sd":     Shape((roi,),     mins=(roi,),      maxs=(roi,)),
            "lap_mean02": Shape((roi,),     mins=(roi,),      maxs=(roi,)),
            "lap_mean04": Shape((roi,),     mins=(roi,),      maxs=(roi,)),
            "lap_sd02":   Shape((roi,),     mins=(roi,),      maxs=(roi,)),
            "lap_sd04":   Shape((roi,),     mins=(roi,),      maxs=(roi,)),
            "r_mean":     Shape((roi, roi), mins=(roi, roi),  maxs=(roi, roi)),
            "r_sd":       Shape((roi, roi), mins=(roi, roi),  maxs=(roi, roi)),
            "roi_means":  Shape((-1, roi),  mins=(TMIN, roi), maxs=(TMAX, roi)),
            "roi_sds":    Shape((-1, roi),  mins=(TMIN, roi), maxs=(TMAX, roi)),
        }[name]
        # fmt: on

    def __str__(self) -> str:
        rois = self.atlas.name if self.atlas is not None else "whole"
        label = f"{self.name} ({rois})"
        sinfo = f"{self.shape_data}"
        location = f"({self.path.relative_to(self.path.parent.parent.parent)})"
        return f"[{label:^25}] {sinfo:<35}    ({location})"

    __repr__ = __str__


ROI_FEATURE_NAMES = [
    "eig_mean",
    "eig_sd",
    "lap_mean02",
    "lap_mean04",
    "lap_sd02",
    "lap_sd04",
    "r_mean",
    "r_sd",
    "roi_means",
    "roi_sds",
]
WHOLE_FEATURE_NAMES = [
    "eig_full",
    "eig_full_c",
    "eig_full_p",
    "eig_full_pc",
]
FEATURES = []
for atlas in ATLASES:  # also include non-atlas-based features
    for fname in ROI_FEATURE_NAMES:
        FEATURES.append(Feature(fname, atlas))
for fname in WHOLE_FEATURE_NAMES:
    FEATURES.append(Feature(fname, None))


if __name__ == "__main__":
    for f in FEATURES:
        print(f)
    # sys.exit()
    for f in FEATURES:
        f.inspect()
