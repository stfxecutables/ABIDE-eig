from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import seaborn as sbn
from ants import image_read
from nibabel.nifti1 import Nifti1Image
from nilearn.image import resample_to_img
from numpy import ndarray
from pandas import DataFrame, Series
from typing_extensions import Literal

DATA = Path(__file__).resolve().parent.parent.parent / "data"
ATLAS = DATA / "atlases/cc200_roi_atlas.nii.gz"
NII = DATA / "niis/CMU_a_0050642_func_minimal.nii.gz"


def add_nii_suffix(nii: Path, suffix: str = "") -> Path:
    extensions = "".join(nii.suffixes)
    end = f"{suffix}.nii.gz"
    return Path(str(nii).replace(extensions, end))


if __name__ == "__main__":
    img = nib.load(str(NII))
    atlas = nib.load(str(ATLAS))
    fixed_atlas = resample_to_img(atlas, img)
    new = Nifti1Image(
        dataobj=fixed_atlas.dataobj,
        affine=fixed_atlas.affine,
        header=fixed_atlas.header,
        extra=fixed_atlas.extra,
    )
    out = str(add_nii_suffix(ATLAS, "CORRECTED"))
    nib.save(new, out)
    print(f"Saved mask to {out}")
