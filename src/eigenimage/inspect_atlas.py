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
MASK_OUT = ATLAS.parent / "MASK.nii.gz"


def add_nii_suffix(nii: Path, suffix: str = "") -> Path:
    extensions = "".join(nii.suffixes)
    end = f"{suffix}.nii.gz"
    return Path(str(nii).replace(extensions, end))


def convert_atlas_to_aligned_mask() -> None:
    img = nib.load(str(NII))
    atlas = nib.load(str(ATLAS))
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
    out = str(add_nii_suffix(MASK_OUT))
    nib.save(new, out)
    print(f"Saved mask to {out}")


def plot_slices(slices: Sequence[Sequence[Any]], img: ndarray, mask: ndarray) -> None:
    assert len(slices) == 9
    fig, axes = plt.subplots(nrows=3, ncols=3)
    for slc, ax in zip(slices, axes.flat):
        ax.imshow(img[slc], cmap="Greys")
        ax.imshow(mask[slc], cmap="inferno", alpha=0.3)
    plt.show()


if __name__ == "__main__":
    # convert_atlas_to_aligned_mask()
    img = nib.load(str(NII)).get_fdata()
    mask = nib.load(str(MASK_OUT)).get_data()

    mean: ndarray = np.mean(img, axis=-1)  # type: ignore
    slices_x = [
        (i, slice(None), slice(None))
        for i in np.linspace(0, img.shape[0] - 1, 9, dtype=int).tolist()
    ]
    slices_y = [
        (slice(None), i, slice(None))
        for i in np.linspace(0, img.shape[1] - 1, 9, dtype=int).tolist()
    ]
    slices_z = [
        (slice(None), slice(None), i)
        for i in np.linspace(0, img.shape[2] - 1, 9, dtype=int).tolist()
    ]
    plot_slices(slices_x, mean, mask)
    plot_slices(slices_y, mean, mask)
    plot_slices(slices_z, mean, mask)
