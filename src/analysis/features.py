import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

# fmt: off
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from src.analysis.preprocess.atlas import Atlas
from src.analysis.preprocess.constants import FEATURES_DIR, T_CROP


@dataclass
class Shape:
    shape: Tuple[int, ...]
    min: int
    max: int


class Feature:
    def __init__(self, name: str) -> None:
        self.name: str = name
        self.path: Path = self.get_path(name)
        self.shape_data: Shape = self.get_shape(name)

    @staticmethod
    def get_path(name: str, atlas: Optional[Atlas] = None) -> Path:
        if atlas is None:
            return Path(FEATURES_DIR / name)
        return Path(FEATURES_DIR / f"{atlas.name}/{name}")

    @staticmethod
    def get_shape(name: str, atlas: Optional[Atlas] = None) -> Shape:
        if atlas is None:
            return {
                "eig_full": Shape((-1,), 77, 295),
                "eig_full_c": Shape((-1,), 77, T_CROP - 1),
                "eig_full_p": Shape((-1,), T_CROP - 1, 315),
                "eig_full_pc": Shape((200,), T_CROP - 1, T_CROP - 1),
            }[name]

        roi = 200 if atlas.name == "cc200" else 392
        return {
            "eig_mean": (roi,),
            "eig_sd": (roi,),
            "lap_mean02": (roi,),
            "lap_mean04": (roi,),
            "lap_sd02": (roi,),
            "lap_sd04": (roi,),
            "r_mean": (),
            "r_sd": (),
            "roi_means": (),
            "roi_sds": (),
        }[name]


class ROIFeature(Feature):
    def __init__(self, name: str, path: Path, shape: Tuple[int, ...], atlas: Atlas) -> None:
        super().__init__(name, path, shape)
        self.atlas: Atlas = atlas


Feature(
    name="eig_full",
)
Feature(
    name="eig_full_c",
)
Feature(
    name="eig_full_p",
)
Feature(
    name="eig_full_pc",
)
ROIFeature(
    name="eig_mean",
)
ROIFeature(
    name="eig_sd",
)
ROIFeature(
    name="lap_mean02",
)
ROIFeature(
    name="lap_mean04",
)
ROIFeature(
    name="lap_sd02",
)
ROIFeature(
    name="lap_sd04",
)
ROIFeature(
    name="r_mean",
)
ROIFeature(
    name="r_sd",
)
ROIFeature(
    name="roi_means",
)
ROIFeature(
    name="roi_sds",
)

{
    "eig_full": (),
    "eig_full_c": (),
    "eig_full_p": (),
    "eig_full_pc": (),
    "eig_mean": (),
    "eig_sd": (),
    "lap_mean02": (),
    "lap_mean04": (),
    "lap_sd02": (),
    "lap_sd04": (),
    "r_mean": (),
    "r_sd": (),
    "roi_means": (),
    "roi_sds": (),
}
