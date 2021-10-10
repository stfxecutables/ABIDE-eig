import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlretrieve

import pandas as pd
from pandas import DataFrame

# fmt: off
CC_CLUSTER = os.environ.get("CC_CLUSTER")
if CC_CLUSTER is not None and (CC_CLUSTER == "niagara"):
    SCRATCH = os.environ["SCRATCH"]
    os.environ["MPLCONFIGDIR"] = str(Path(SCRATCH) / ".mplconfig")
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

from src.analysis.preprocess.atlas import Atlas

SUFFIX = "_f16_subsample" if CC_CLUSTER is None else ""
DATA = ROOT / "data"
NIIS = DATA / f"nii_cpac{SUFFIX}"
FEATURES_DIR = DATA / f"features_cpac{SUFFIX}"

ROIS = DATA / "rois"
if not ROIS.exists():
    os.makedirs(ROIS, exist_ok=True)
    os.makedirs(ROIS / "ctrl", exist_ok=True)
    os.makedirs(ROIS / "autism", exist_ok=True)
EIGS = DATA / "eigs"  # for normalizing
SUBJ_DATA = DATA / "Phenotypic_V1_0b_preprocessed1.csv"
EIGIMGS = DATA / "eigimgs"

# NOTE!!! CC_400 only actually has 392 ROIS...
ATLAS_DIR = DATA / "atlases"
ATLAS_400 = ATLAS_DIR / "cc400_roi_atlas_ALIGNED.nii.gz"
LEGEND_400 = ATLAS_DIR / "CC400_ROI_labels.csv"
ATLAS_200 = ATLAS_DIR / "cc200_roi_atlas_ALIGNED.nii.gz"
LEGEND_200 = ATLAS_DIR / "CC200_ROI_labels.csv"
CC200 = Atlas("cc200", ATLAS_200, LEGEND_200)
CC400 = Atlas("cc400", ATLAS_400, LEGEND_400)
ATLASES = [CC200, CC400]

MASK = ATLAS_DIR / "MASK.nii.gz"

T_CROP = 203
