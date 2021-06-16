from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
    cast,
    no_type_check,
)

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
from ants import image_read
from nibabel.nifti1 import Nifti1Image
from nilearn.image import resample_to_img
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

DATA = Path(__file__).resolve().parent.parent.parent / "data"
ATLAS_DIR = Path(__file__).resolve().parent
ATLASES = ATLAS_DIR.rglob("*atlas.nii.gz")
NII = ATLAS_DIR.parent / "niis/CMU_a_0050642_func_minimal.nii.gz"
CC = ATLAS_DIR / "cc200_roi_atlas.nii.gz"


def add_nii_suffix(nii: Path, suffix: str = "") -> Path:
    extensions = "".join(nii.suffixes)
    end = f"{suffix}.nii.gz"
    return Path(str(nii).replace(extensions, end))

def make_master_mask() -> None:
    # only the Craddock (CC) atlas is of any serious quality for masking
    img = nib.load(str(NII))
    atlas = nib.load(str(CC))
    fixed_atlas = resample_to_img(atlas, img)
    data = fixed_atlas.get_data()
    binary = (data != 0).astype(np.int8)
    new = Nifti1Image(
        dataobj=binary,
        affine=fixed_atlas.affine,
        header=fixed_atlas.header,
        extra=fixed_atlas.extra,
    )
    new.set_data_dtype(np.int8)
    out = str(add_nii_suffix(ATLAS_DIR / "MASK.nii.gz"))
    nib.save(new, out)
    print(f"Saved mask to {out}")

def realign_atlases() -> None:
    img = nib.load(str(NII))
    for path in ATLASES:
        atlas = nib.load(str(path))
        fixed_atlas = resample_to_img(atlas, img)
        new = Nifti1Image(
            dataobj=fixed_atlas.dataobj,
            affine=fixed_atlas.affine,
            header=fixed_atlas.header,
            extra=fixed_atlas.extra,
        )
        out = str(add_nii_suffix(path, "_ALIGNED"))
        nib.save(new, out)
        print(f"Saved corrected atlas to {out}")


if __name__ == "__main__":
    realign_atlases()
