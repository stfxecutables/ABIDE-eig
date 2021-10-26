import os
from pathlib import Path
from typing import List

from src.constants.environment import CC_CLUSTER, SLURM_TMPDIR


def ensure_path(path: Path) -> Path:
    if not path.exists():
        os.makedirs(path, exist_ok=True)
    return path


ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / "data"
DEEP = DATA / "deep"
EIGS = DATA / "eigs"
NIIS = DATA / "niis"
ROIS = DATA / "rois"
SEQS = DATA / "seqs"
EIGIMGS = DATA / "eigimgs"  # raw eigenimages
CKPT_PATH = ensure_path(DATA / "ckpts")

ATLAS_DIR = DATA / "atlases"
MASK = ATLAS_DIR / "MASK.nii.gz"  # perhaps only want loss on a mask
SUBJ_DATA = DATA / "Phenotypic_V1_0b_preprocessed1.csv"
SCRIPT_OUTDIR = ensure_path(ROOT / "job_scripts")

DEEP_FMRI = DEEP / "fmri"
DEEP_EIGIMG = DEEP / "eigimg"
FULL_DEEP_FMRI = DATA / ("nii_cpac" if CC_CLUSTER is not None else "nii_cpac_f16_subsample")
FULL_DEEP_EIGIMG = DEEP / "nii_cpac_eigimg"
# Not all images convert to eigenimg of same dims, so we only use fMRI
# images that we could compute comparable eigimgs for.
PREPROC_EIG: List[Path] = sorted(DEEP_EIGIMG.rglob("*.npy"))
PREPROC_FMRI: List[Path] = [DEEP_FMRI / str(p.name).replace("_eigimg", "") for p in PREPROC_EIG]
FULL_PREPROC_EIG: List[Path] = sorted(FULL_DEEP_EIGIMG.rglob("*.npy"))
# FULL_PREPROC_FMRI = [FULL_DEEP_FMRI / str(p.name).replace("_eigimg", "") for p in FULL_PREPROC_EIG]  # noqa
FULL_PREPROC_FMRI: List[Path] = sorted(FULL_DEEP_FMRI.rglob("*.nii.gz"))
