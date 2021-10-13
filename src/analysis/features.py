from __future__ import annotations  # isort:skip # noqa

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sbn
from matplotlib.lines import Line2D
from numpy import ndarray
from scipy.stats import boxcox, norm, yeojohnson
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

# fmt: off
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

# HACK for local testing
if os.environ.get("CC_CLUSTER") is None:
    os.environ["CC_CLUSTER"] = "home"

from data.download_cpac_1035 import download_csv

from src.analysis.preprocess.atlas import Atlas
from src.analysis.preprocess.constants import ATLASES, FEATURES_DIR, T_CROP

Arrays = List[ndarray]
NormMethod = Literal["f-minmax", "s-minmax", "f-sd", "s-sd", "yj"]

SUBJ_DATA = download_csv().loc[:, ["fname", "DX_GROUP"]]
# fix idiotic default of 1 == ASD, 2 == TD, so that now
# 1 = autism, 0 = TD (typical development, control)
SUBJ_DATA.DX_GROUP = SUBJ_DATA.DX_GROUP.apply(lambda x: 2 - x)
SUBJ_DATA.index = pd.Index(SUBJ_DATA.fname.values, name="fname")
SUBJ_DATA.drop(columns="fname", inplace=True)

COLORS = {
    "All subjects": "black",
    "All Subjects": "black",
    "ASD": "#ffa514",
    "TD": "#146eff",
}
TITLES = {
    "roi_means": "ROI mean signals",
    "roi_sds": "ROI standard deviation signals",
    "r_mean": "Correlations of ROI mean signals",
    "r_sd": "Correlations of ROI standard deviation signals",
    "eig_mean": "Eigenvalues of correlations of ROI mean signals",
    "eig_sd": "Eigenvalues of standard deviations of ROI mean signals",
    "lap_mean02": "Eigenvalues of Laplacian of thresholded corrs. of ROI means (T=0.2)",
    "lap_mean04": "Eigenvalues of Laplacian of thresholded corrs. of ROI means (T=0.4)",
    "lap_sd02": "Eigenvalues of Laplacian of thresholded corrs. of ROI sds (T=0.2)",
    "lap_sd04": "Eigenvalues of Laplacian of thresholded corrs. of ROI sds (T=0.4)",
    "eig_full_pc": "Eigenvalues of all voxel-voxel correlations (padded and cropped)",
    "eig_full": "Eigenvalues of all voxel-voxel correlations",
    "eig_full_c": "Eigenvalues of all voxel-voxel correlations (cropped)",
    "eig_full_p": "Eigenvalues of all voxel-voxel correlations (padded)",
}
LEGEND_LOC = {
    "roi_means": {"top": None, "bottom": None},
    "roi_sds": {"top": None, "bottom": None},
    "r_mean": {"top": None, "bottom": None},
    "r_sd": {"top": None, "bottom": None},
    "eig_mean": {"top": "top left", "bottom": "top right"},
    "eig_sd": {"top": "", "bottom": ""},
    "lap_mean02": {"top": "bottom right", "bottom": "top left"},
    "lap_mean04": {"top": "bottom right", "bottom": "top left"},
    "lap_sd02": {"top": "bottom right", "bottom": "top left"},
    "lap_sd04": {"top": "top left", "bottom": "top right"},
    "eig_full_pc": {"top": "top left", "bottom": "top right"},
    "eig_full": {"top": "top left", "bottom": "top right"},
    "eig_full_c": {"top": "top left", "bottom": "top right"},
    "eig_full_p": {"top": "top left", "bottom": "top right"},
}

FILETYPE = ".png"
DPI = 300
INSPECT_OUTDIR = ROOT / "results/feature_plots"
os.makedirs(INSPECT_OUTDIR, exist_ok=True)


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
            arrs = self._normalize(arrs, method="eig")
        if not stack:
            return arrs, y
        x = self._stack_subjects(arrs)
        return x, y

    def compare_normalizations(self) -> None:
        """Compare eigenvalue normalization techniques.

        Compare: 11 = 2*6
            s-mx, f-mx, s-sd, f-sd, yj,
            f-mx+yj, yj+f-mx,  f-sd+yj, yj+f-sd,  yj+s-mx, yj+s-sd   (f first, mx first)

        Notes
        -----
        When all options are plotted together like this, it becomes clear the only
        real options are featurewise minmax or featurewise standardization, or just
        Yeo-Johnson.

        YJ maps all but the top quarter or third of eigenindices to zero, so embodies a prior that
        only the large eigenvalues really matter. However it also smooths them out a lot, so maybe
        also encodes the assumption eigensignals are noisy...

        Featurewise MinMax *in theory* emobodies a prior that all eigenindices are equally important,
        but we can see from the plots it effectively renders central eigen-indices constant. This is
        because central eigenvalues are well-behaved / less variable than the extreme (small or large)
        values. So practically, featurewise min-max normalization actually embeds a prior that the
        spectrum tails are most important.

        Featurewise "medianization" clearly has the practical effect of keeping variation only in in
        the central / bulk eigenvalues. Presumably, central eigenindices are have an average greater
        than their median, and are not crazily skewed, so divding by the median gives values in e.g.
        [0, 5] or so. By contrast, it must be that extreme eigenindices are skewed extremely heavily
        in the opposite way, since medianizing sends these all to near zero. BUT THIS ONLY WORKS for
        eigenvalues of ROI means.

        # Summary

        | Feature  | Atlas | ftwise max | ftwise std | ftwise med | Yeo-Johnson |  clip sd | subj max | subj std |
        ------------------------------------------------------------------------------------------------------------
        | eig_mean |  200  |     Y      |     Y      |  p=25-33   |     ?       |    NO    |    NO    |    NO    |
        | eig_mean |  400  |     Y      |     Y      |  p=25-33   |     ?       |    NO    |    NO    |    NO    |
        |  eig_sd  |  200  |     Y      |     Y      |    NO      |     Y       |    NO    |    NO    |    NO    |
        |  eig_sd  |  400  |     Y      |     Y      |    NO      |     Y       |    NO    |    NO    |    NO    |
        | eig_full |       |     Y      |     ?      |    NO      |     ?       | [-5,20]? |    NO    |    NO    |
        | lap_mn02 |  200  |     Y      |     ?      |     Y      |     NO      |    NO    |    Y     |    NO    |
        | lap_mn02 |  400  |     Y      |     Y      |     Y      |     NO      |    NO    |    Y     |    NO    |
        | lap_mn04 |  200  |     Y      |     ?      |    NO      |     NO      |    NO    |    Y     |    NO    |
        | lap_mn04 |  400  |     Y      |     ?      |    NO      |     NO      |    NO    |    Y     |    NO    |
        | lap_sd02 |  200  |     Y      |     ?      |     Y      |     NO      | [-7.5,5]Y|    Y     |    NO    |
        | lap_sd02 |  400  |     Y      |     Y      |     Y      |     NO      |  [-5,5]Y |    Y     |    NO    |
        | lap_sd04 |  200  |     Y      |     Y      |    NO      |     NO      | [-5,7.5]Y|    Y     |    NO    |
        | lap_sd04 |  400  |     Y      |     Y      |    NO      |     NO      | [-5,10]Y |    Y     |    NO    |



        Subject-level standardization just doesn't work AT ALL (though probably does for laplacian), so we don't even bother.
        """
        arrs, y = self.load(normalize=False, stack=False)
        fig, axes = plt.subplots(nrows=4, ncols=6)

        # one-shots
        normed = self._minmax_1d(arrs, featurewise=False)
        self.plot_eig_feature(axes[0][0], axes[1][0], normed, y, "subj min-max")
        axes[0][0].set_yscale("log")
        normed = self._minmax_1d(arrs, featurewise=True)
        self.plot_eig_feature(axes[0][1], axes[1][1], normed, y, "feat min-max")
        axes[0][1].set_yscale("linear")
        normed = self._medianize_1d(arrs, p=33.33, featurewise=False)
        self.plot_eig_feature(axes[0][2], axes[1][2], normed, y, "subj medianize")
        axes[0][2].set_yscale("log")
        normed = self._standardize_1d(arrs, featurewise=True)
        self.plot_eig_feature(axes[0][3], axes[1][3], normed, y, "feat standardize")
        axes[0][3].set_yscale("linear")
        normed = self._yj(arrs)
        self.plot_eig_feature(axes[0][4], axes[1][4], normed, y, "Yeo-Johnson")
        axes[0][4].set_yscale("linear")
        normed = self._medianize_1d(arrs, p=33.33, featurewise=True)
        self.plot_eig_feature(axes[0][5], axes[1][5], normed, y, "feat medianize")
        axes[0][5].set_yscale("log")
        # fig.delaxes(axes[0][5])
        # fig.delaxes(axes[1][5])

        # combination
        normed = self._yj(self._minmax_1d(arrs, featurewise=True))
        self.plot_eig_feature(axes[2][0], axes[3][0], normed, y, "feat min-max, YJ")
        axes[2][0].set_yscale("linear")
        normed = self._minmax_1d(self._yj(arrs), featurewise=True)
        self.plot_eig_feature(axes[2][1], axes[3][1], normed, y, "YJ, feat min-max")
        axes[2][1].set_yscale("linear")
        normed = self._yj(self._standardize_1d(arrs, featurewise=True))
        self.plot_eig_feature(axes[2][2], axes[3][2], normed, y, "feat standardize, YJ")
        axes[2][2].set_yscale("linear")
        normed = self._standardize_1d(self._yj(arrs), featurewise=True)
        self.plot_eig_feature(axes[2][3], axes[3][3], normed, y, "YJ, feat standardize")
        axes[2][3].set_yscale("linear")
        normed = self._minmax_1d(self._yj(arrs), featurewise=False)
        self.plot_eig_feature(axes[2][4], axes[3][4], normed, y, "YJ, subj min-max")
        axes[2][4].set_yscale("linear")
        normed = self._standardize_1d(self._yj(arrs), featurewise=False)
        self.plot_eig_feature(axes[2][5], axes[3][5], normed, y, "YJ, subj standardize")
        axes[2][5].set_yscale("linear")

        for ax in axes.ravel():
            ax.set_xlabel("")
            ax.set_ylabel("")
        fig.set_size_inches(h=16, w=16)
        atlas = f" ({self.atlas.name.upper()} atlas)" if self.atlas is not None else ""
        fig.suptitle(
            f"{TITLES[self.name]}{atlas} - x/y axes = (feat. index / feat. value) OR (feat. value / count)"
        )
        fig.subplots_adjust(top=0.931, bottom=0.06, left=0.043, right=0.99, hspace=0.3, wspace=0.2)
        plt.show()

    def _normalize(
        self,
        arrs: Arrays,
        method: List[NormMethod],
        featurewise: bool = True,
        log: bool = False,
    ) -> Arrays:
        """Normalize feature using feature-relevant method. In most cases just subject-wise minmax.

        Notes
        -----
        Most promising methods:

        [Yeo-Johnson]
            - better than boxcox
        [featurewise minmax + YJ]
            - dist still highly skewed, not in [0, 1]
            - does not work at all for SD eigs
            - not bad for full eigs, but still skewed
        [YJ + featurewise minmax]
            - extremely good for mean ROI 200, 400 eigs
            - very promising for SD ROI 200
            - quite promising for SD ROI 400 as well
            - maybe okay for full eigs, certainly fixes distribution
        [YJ + subjectwise minmax]
            - decent for mean ROI 200, 400 (usual binarization)
            - same as line above for SD ROI 200, 400
            - maybe not so great for full eigs
        [subjectwise minmax + YJ] [NO]
            - seems to cause way too much shrinking in mean ROI eigs
            - interesting effect on SD roi eigs, not sure desirable though
            - huge amount of zero-features in full eigs

        [featurewise-sd + YJ]
            - mostly FANTASTIC for mean ROI 200, 400 eigs, in about [-3, 3]
            - pretty good for sd ROI 200, 400 eigs, in about [-5, 5], some crazy positive outliers (>10) in 200 case
            - OK for full eigs, makes giant hole (all zeros) in middle eigenvalues (e.g. at index 100)
        [YJ + featurewise-sd] [NO, distortions too severe, except maybe for sd200 eigs]
            - strange effects in lower eigenvalues (<75) for mean ROI 200 eigs (values mostly in [-5, 5])
            - similar *very* strange effect for index up to ~275 in mean ROI 400 eigs (values most in [-5, 5])
            - good, quite promising for sd 200 ROI eigs
            - similar strange banding effects up ~ index 100 on CC400 atlas
            - not bad for full eigenvalues, dead zone at index 200, would need to clip to [-5, 10] or so still to
              kill some large outliers
        [YJ + subjectwise-sd]
            - lower indexes up to 100 constant

        [boxcox]
        - Maps a *tonne* of eigenindexes to constants, so seems
            to be destroying a lot of info BUT gives non-sparse data in [0, 1]
        - does not work on eigenvalues of ROI sd signals
        - does not put full eigenvalues in [0, 1] (further minmax would be needed)
        [featurewise minmax + box]
        - promising, but requires another minmax to put ROI mean eigs in [0, 1]
        - same for ROI sd eigs (though some craziness appears here too)
        - also works for full eigenvalues
        [featurewise minmax + box + subjectwise-minmax]
        - definitely very promising, though very crazy features in all cases
        [featurewise minmax + box + featurewise-minmax] [BEST SO FAR]
        - excellent for mean ROI eigs, probably ideal
        - still strong skew for SD ROI eigs in CC200 case
        - quite good for full eigs too
        [box + subjectwise-minmax]
        - kind of binarizes the features to all near-zero or near-one for mean eigs
        - similar for SD eigs, though flatter distribution
        - probably also okay for full eigs
        [featurewise-box]
        - just no, doesn't work at all, ill-advised because *within* a feature, dist is not
            really exponential

        [subjectwise minmax + box]
        - NO destroys all features
        """

        if "eig_" in self.name:  # x is list of 1D
            # arrs = featurewise_1d_minmax(arrs)
            # return box(arrs)
            # arrs = subjectwise_1d_minmax(arrs)
            # arrs = featurewise_1d_minmax(arrs)
            # arrs = box(arrs, featurewise=True)
            # arrs = box(arrs)
            # arrs = subjectwise_1d_minmax(arrs)
            # arrs = self._minmax_1d(arrs)
            # arrs = self._standardize_1d(arrs, featurewise=True)
            arrs = self._yj(arrs)
            arrs = self._standardize_1d(arrs, featurewise=False)
            # arrs = featurewise_1d_minmax(arrs)
            # arrs = yj(arrs)
            return arrs
            return kmeans(arrs)

        else:
            return arrs

    def _stack_subjects(self, arrs: Arrays, unified_size: int = 200) -> ndarray:
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

    def describe(self) -> None:
        """Describe crucial percentile and range info, for normalization decisions"""
        shape = self.shape_data.shape
        if len(shape) == 1:
            self.describe_1d_feature()
            return
        if len(shape) == 2:
            self.describe_2d_feature()

    def inspect(self, show: bool = True, normalize: bool = False) -> None:
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
            self.plot_1d_feature(show, normalize)
            return
        if len(shape) == 2:
            self.plot_2d_feature(show, normalize)
        # now either we have a matrix, or actual waveforms

    def describe_1d_feature(self) -> None:
        x, y = self.load(normalize=False, stack=False)

    def plot_1d_feature(self, show: bool, normalize: bool) -> None:
        x, y = self.load(normalize=normalize, stack=True)
        x_asd, x_td = x[y == 0], x[y == 1]
        fig, axes = self._plot_setup()

        info_all = self.plot_hist(axes[1][0], x, "All subjects")
        info_asd = self.plot_hist(axes[1][1], x_asd, "ASD")
        info_td = self.plot_hist(axes[1][2], x_td, "TD")
        # self.plot_curves(axes[0][0], x, "All subjects")
        self.plot_curves(axes[0][0], x_asd, "ASD", color="#ffa514")
        self.plot_curves(axes[0][0], x_td, "TD", color="#146eff")
        handles = [
            Line2D([0], [0], color="#ffa514", lw=0.75),
            Line2D([0], [0], color="#146eff", lw=0.75),
        ]
        labels = ["ASD", "TD"]
        axes[0][0].legend(handles, labels)
        axes[0][0].set_title("All Subjects")
        self.plot_curves(axes[0][1], x_asd, "ASD")
        self.plot_curves(axes[0][2], x_td, "TD")
        fig.text(x=0.16, y=0.05, s=info_all)
        fig.text(x=0.45, y=0.05, s=info_asd)
        fig.text(x=0.72, y=0.05, s=info_td)
        fig.subplots_adjust(top=0.91, bottom=0.22, left=0.125, right=0.9, hspace=0.32, wspace=0.2)
        fig.set_size_inches(h=8, w=16)
        atlas = f" ({self.atlas.name.upper()} atlas)" if self.atlas is not None else ""
        fig.suptitle(f"{TITLES[self.name]}{atlas}")
        if show:
            plt.show()
        else:
            fig.savefig(INSPECT_OUTDIR / f"{self.name}_{atlas.replace(' ', '')}{FILETYPE}", dpi=DPI)
        plt.close()

    def plot_eig_feature(
        self,
        ax_curve: plt.Axes,
        ax_hist: plt.Axes,
        arrs: ndarray,
        y: ndarray,
        title: str,
        legend: bool = False,
    ) -> None:
        x = self._stack_subjects(arrs)
        x_asd, x_td = x[y == 0], x[y == 1]
        self.plot_hist(ax_hist, x, "All subjects")
        ax_hist.get_legend().remove()
        ax_hist.set_title("")
        self.plot_curves(ax_curve, x_asd, "ASD", color="#ffa514")
        self.plot_curves(ax_curve, x_td, "TD", color="#146eff")
        if legend:
            handles = [
                Line2D([0], [0], color="#ffa514", lw=0.75),
                Line2D([0], [0], color="#146eff", lw=0.75),
            ]
            labels = ["ASD", "TD"]
            ax_curve.legend(handles, labels)
        ax_curve.set_title(title)

    def plot_2d_feature(self, show: bool, normalize: bool) -> None:
        x, y = self.load(normalize=normalize, stack=True)
        x_asd, x_td = x[y == 0], x[y == 1]
        fig, axes = self._plot_setup()
        atlas = f" ({self.atlas.name} atlas)" if self.atlas is not None else ""
        if "r_" in self.name:
            # grab upper triangle of matrix, flatten, sort features by variance,
            # make x = feature index, y = feature value,
            # do a big scatter plot with transparency
            self.scatter_2d(axes[0][0], x, "All Subjects")
            self.scatter_2d(axes[0][1], x_asd, "ASD")
            self.scatter_2d(axes[0][2], x_td, "TD")
            self.pseudo_sequence_2d(axes[1][0], x, "All Subjects")
            self.pseudo_sequence_2d(axes[1][1], x_asd, "All Subjects")
            self.pseudo_sequence_2d(axes[1][2], x_td, "All Subjects")
            fig.suptitle(
                f"{TITLES[self.name]}{atlas}\n"
                "Feature Percentiles (top) and Feature means (bottom) across subjects"
            )
        elif "roi_" in self.name:
            # on top, plot staggered signals colored by spectral pallete
            # on bottom plot the feature image (as.imshow), time as x-axis
            feat_sds = np.std(x, axis=0, ddof=1)
            space = 2 * np.max(feat_sds)
            self.maxmin_signals(axes[0][0], x, space, "All Subjects")
            self.maxmin_signals(axes[0][1], x_asd, space, "ASD")
            self.maxmin_signals(axes[0][2], x_td, space, "TD")
            self.imshow(axes[1][0], x, "All Subjects")
            self.imshow(axes[1][1], x_asd, "ASD")
            self.imshow(axes[1][2], x_td, "TD")
            fig.suptitle(
                f"{TITLES[self.name]}{atlas}\n"
                "ROI max and min with colour indicating feature index (top) and Feature mean image (bottom) across subjects"
            )
        adjust = dict(bottom=0.1, hspace=0.4) if "200" in atlas else dict(hspace=0.25)
        fig.subplots_adjust(**adjust)
        fig.set_size_inches(h=8, w=16)
        if show:
            plt.show()
        else:
            fig.savefig(INSPECT_OUTDIR / f"{self.name}_{atlas.replace(' ', '')}{FILETYPE}", dpi=DPI)
        plt.close()
        # just scatter plot

    @staticmethod
    def imshow(ax: plt.Axes, x: ndarray, title: str) -> None:
        # t is first dimension after batch
        cmap = sbn.color_palette("icefire", as_cmap=True)
        img = np.mean(x, axis=0)
        vmin, vmax = np.percentile(img, [0, 100])
        ax.matshow(img, cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.set_ylabel("Time")
        ax.set_xlabel("Feature index")

    @staticmethod
    def maxmin_signals(ax: plt.Axes, x: ndarray, space: float, title: str) -> None:
        # t is first dimension after batch
        # palette = sbn.color_palette("icefire", n_colors=x.shape[2])
        palette = sbn.color_palette("flare", n_colors=x.shape[2])
        t = list(range(x.shape[1]))
        for k in range(x.shape[2]):  # x[i, :, k] is timeseries
            signal = np.max(x[:, :, k], axis=0) + k * space  # stagger heights
            signal = np.min(x[:, :, k], axis=0) + k * space  # stagger heights
            # signal = x[i, :, k]  # stagger heights
            ax.plot(t, signal, color=palette[k], lw=0.4, alpha=0.9)
        ax.set_title(title)
        ax.set_xlabel("time")
        ax.set_ylabel("Feature values (staggered)")

    @staticmethod
    def scatter_2d(ax: plt.Axes, x: ndarray, title: str) -> None:
        tri_idx = np.tril_indices_from(x[0], k=1)
        mask = np.zeros_like(x[0], dtype=bool)
        mask[tri_idx] = True
        x = x[:, mask]
        variances = np.std(x, axis=0, ddof=1)
        sort_idx = np.argsort(variances)
        x = x[:, sort_idx]
        # It is too much to scatterplot each subject.Instead, let's just plot some percentiles
        ps = np.linspace(0, 100, 32)
        percentiles = np.percentile(x, ps, axis=0)
        palette = sbn.color_palette("icefire", n_colors=len(ps), as_cmap=False)
        idx = list(range(x.shape[1]))
        for i in list(range(percentiles.shape[0])):
            ax.scatter(idx, percentiles[i], color=palette[i], s=0.2, alpha=0.1)
        ax.set_title(title)
        ax.set_xlabel("Feature index (variance sorted)")
        ax.set_ylabel("Feature percentile values")

    @staticmethod
    def pseudo_sequence_2d(ax: plt.Axes, x: ndarray, title: str) -> None:
        tri_idx = np.tril_indices_from(x[0], k=1)
        mask = np.zeros_like(x[0], dtype=bool)
        mask[tri_idx] = True
        x = x[:, mask]
        means = np.mean(x, axis=0)  # feature means
        sort_idx = np.argsort(means)
        x = x[:, sort_idx]
        idx = range(x.shape[1])
        for i in range(x.shape[0]):
            ax.plot(idx, x[i], lw=0.1, alpha=0.1, color=COLORS[title])
        ax.set_title(title)
        ax.set_xlabel("Feature index (mean sorted)")
        ax.set_ylabel("Feature values")

    @staticmethod
    def _plot_setup() -> Tuple[plt.Figure, plt.Axes]:
        fig, axes = plt.subplots(nrows=2, ncols=3, sharex=False, sharey=False)
        axes[1][1].sharex(axes[1][0])
        axes[1][2].sharex(axes[1][0])
        axes[1][1].sharey(axes[1][0])
        axes[1][2].sharey(axes[1][0])

        axes[0][1].sharey(axes[0][0])
        axes[0][2].sharey(axes[0][0])
        axes[0][1].sharex(axes[0][0])
        axes[0][2].sharex(axes[0][0])
        return fig, axes

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
        # alpha = 0.2 if color == "black" else 0.3
        alpha = 0.1
        for i in range(x.shape[0]):
            lab = label if (i == 0) else None
            ax.plot(x[i], color=color, alpha=alpha, lw=0.5, label=lab)
        ax.set_xlabel("Feature index")
        ax.set_ylabel("Feature value")
        # ax.set_xscale("log")
        if "eig_" in self.name:
            ax.set_yscale("log")
        ax.set_title(label)

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

    @staticmethod
    def _box(arrs: Arrays, featurewise: bool = False) -> Arrays:
        """Box-Cox "normalization"."""
        normed = []
        for x in arrs:
            x[np.isnan(x)] = 0
            if np.std(x) <= 0:
                normed.append(x)
                continue
            m = np.min(x) + 1
            a = x + np.abs(m) + 1 if m <= 1 else x
            normed.append(boxcox(a)[0])
        return normed

    def _yj(self, arrs: Arrays) -> Arrays:
        """Yeo-Johnson normalization. Does not have annoying constraints of Box-Cox."""
        normed = []
        for x in arrs:
            x[np.isnan(x)] = 0
            if np.std(x) <= 0:
                normed.append(x)
                continue
            normed.append(yeojohnson(x)[0])
        return normed

    def _minmax_1d(self, arrs: Arrays, featurewise: bool = True) -> Arrays:
        """Simple MinMax Normalization / feature scaling."""

        def _featurewise(arrs: Arrays) -> ndarray:
            x = self._stack_subjects(arrs)
            mxs = np.max(x, axis=0)
            mns = np.min(x, axis=0)
            normed = []
            for i, arr in enumerate(arrs):
                f = len(arr)  # only use last f (n feature) means since front-pad
                a = (arr - mns[-f:]) / (mxs[-f:] - mns[-f:])
                normed.append(a)
            return normed

        def _subjectwise(arrs: Arrays) -> Arrays:
            normed = []
            for arr in arrs:
                mx = np.max(arr)
                mn = np.min(arr)
                x = (arr - mn) / (mx - mn)
                normed.append(x)
            return normed

        res: List[ndarray] = _featurewise(arrs) if featurewise else _subjectwise(arrs)
        return res

    def _standardize_1d(self, arrs: Arrays, featurewise: bool = True) -> Arrays:
        def _featurewise(arrs: Arrays) -> ndarray:
            x = self._stack_subjects(arrs)
            m = np.mean(x, axis=0)
            sd = np.std(x, axis=0, ddof=1)
            normed = []
            for i, arr in enumerate(arrs):
                f = len(arr)  # only use last f (n feature) means since front-pad
                a = (arr - m[-f:]) / sd[-f:]
                normed.append(a)
            return normed

        def _subjectwise(arrs: Arrays) -> Arrays:
            normed = []
            for arr in arrs:
                m = np.mean(arr)
                sd = np.std(arr, ddof=1)
                x = (arr - m) / sd
                normed.append(x)
            return normed

        res: List[ndarray] = _featurewise(arrs) if featurewise else _subjectwise(arrs)
        return res

    def _medianize_1d(self, arrs: Arrays, featurewise: bool = True, p: float = 25) -> Arrays:
        def _featurewise(arrs: Arrays) -> ndarray:
            x = self._stack_subjects(arrs)
            mn, m, mx = np.nanpercentile(x, [p, 50, 100 - p], axis=0)
            sd = mx - mn
            normed = []
            for i, arr in enumerate(arrs):
                f = len(arr)  # only use last f (n feature) means since front-pad
                a = (arr - m[-f:]) / sd[-f:]
                normed.append(a)
            return normed

        def _subjectwise(arrs: Arrays) -> Arrays:
            normed = []
            for arr in arrs:
                mn, m, mx = np.nanpercentile(arr, [p, 50, 100 - p], axis=0)
                sd = mx - mn
                x = (arr - m) / sd
                normed.append(x)
            return normed

        res: List[ndarray] = _featurewise(arrs) if featurewise else _subjectwise(arrs)
        return res

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


def create_feature_list() -> List[Feature]:
    UNSORTED: List[Feature] = []
    for atlas in ATLASES:  # also include non-atlas-based features
        for fname in ROI_FEATURE_NAMES:
            UNSORTED.append(Feature(fname, atlas))
    for fname in WHOLE_FEATURE_NAMES:
        UNSORTED.append(Feature(fname, None))

    features = []
    for name in TITLES:
        for f in UNSORTED:
            if f.name == name:
                features.append(f)
    return features


FEATURES: List[Feature] = create_feature_list()


def call(f: Feature) -> None:
    f.inspect(show=False)


if __name__ == "__main__":
    mpl.style.use("fast")
    f: Feature
    for f in FEATURES:
        print(f)
        if "eig_" in f.name or "lap" in f.name:
            f.compare_normalizations()
    sys.exit()
    process_map(call, FEATURES)
