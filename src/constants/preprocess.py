from src.analysis.preprocess.atlas import Atlas
from src.constants.environment import CC_CLUSTER
from src.constants.paths import DATA

SUFFIX = "_f16_subsample" if CC_CLUSTER is None else ""
NIIS = DATA / f"nii_cpac{SUFFIX}"
FEATURES_DIR = DATA / f"features_cpac{SUFFIX}"

# NOTE!!! CC_400 only actually has 392 ROIS...
ATLAS_DIR = DATA / "atlases"
ATLAS_400 = ATLAS_DIR / "cc400_roi_atlas_ALIGNED.nii.gz"
LEGEND_400 = ATLAS_DIR / "CC400_ROI_labels.csv"
ATLAS_200 = ATLAS_DIR / "cc200_roi_atlas_ALIGNED.nii.gz"
LEGEND_200 = ATLAS_DIR / "CC200_ROI_labels.csv"
CC200 = Atlas("cc200", ATLAS_200, LEGEND_200)
CC400 = Atlas("cc400", ATLAS_400, LEGEND_400)
ATLASES = [CC200, CC400]

T_CROP = 203
