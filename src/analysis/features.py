from __future__ import annotations  # isort:skip # noqa

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# fmt: off
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from src.analysis.preprocess.atlas import Atlas
from src.analysis.preprocess.constants import ATLASES, FEATURES_DIR, T_CROP


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
