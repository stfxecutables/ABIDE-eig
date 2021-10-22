# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

import os
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple, cast

import nibabel as nib
import numpy as np
from numpy import ndarray
from tqdm.contrib.concurrent import process_map

from src.constants.environment import SLURM_TMPDIR
from src.constants.paths import DEEP, EIGIMGS, EIGS, MASK, NIIS

"""We need to use fMRI (non-eigimg) files only for subjects where we could
compute the eigenimage of the same shapes. For efficiency, we also need to
trim these fMRI files to the last 175 timepoints, and apply the mask, and
save these, which will allow all data to be loaded into memory and much
more efficient processing. We may also want to trim down black space after
zero-masking to further simplify file sizes and produce even shapes with
more reliable behaviour.
"""

PREPROCESSED = SLURM_TMPDIR / "data" if SLURM_TMPDIR is not None else None
DEEP_FMRI = DEEP / "fmri"
DEEP_EIGIMG = DEEP / "eigimg"
for dir in [DEEP_FMRI, DEEP_EIGIMG]:
    if not dir.exists():
        os.makedirs(dir, exist_ok=True)

T = 175
EIGIMG_SHAPE = (61, 73, 61, T)

Cropper = Tuple[slice, slice, slice, slice]
Transform = Callable[[ndarray], ndarray]


@dataclass
class PreprocArgs:
    nii: Path
    mask: ndarray
    cropper: Cropper
    normalize: bool
    transforms: List[Transform]


def get_mask_bounds(mask: ndarray) -> Cropper:
    # np.max(mask, axis=(1,2)) is True (1) where there is actual image data, so np.where gives:
    # Hs = (array([ 7,  8,  9, 10, 11, 12, ..., 50, 51, 52, 53]),)[0]
    # and so mask[Hs[0]:Hs[-1]+1] is the slice with actual data
    Hs = np.where(np.max(mask, axis=(1, 2)))[0]
    Ws = np.where(np.max(mask, axis=(0, 2)))[0]
    Ds = np.where(np.max(mask, axis=(0, 1)))[0]
    Hmin, Hmax = Hs[0], Hs[-1] + 1
    Wmin, Wmax = Ws[0], Ws[-1] + 1
    Dmin, Dmax = Ds[0], Ds[-1] + 1
    cropper = (slice(Hmin, Hmax), slice(Wmin, Wmax), slice(Dmin, Dmax), slice(None))
    return cropper


def crop_to_bounds(nii: Path, mask: ndarray, cropper: Cropper) -> ndarray:
    img: ndarray = nib.load(str(nii)).get_fdata()
    img[~mask] = 0
    img = img[cropper]
    # now crop time
    img = img[:, :, :, -T:]
    if img.shape[-1] != T:
        raise ValueError(f"Image {nii.name} too short for inclusion")
    return img


def normalize(src: Path, cropped: ndarray) -> ndarray:
    """Requires a cropped array. If an eigenimage, normalize by division of the full
    eigenvales. If the registered, minimally-preprocessed fMRI, divide out the mean
    signal."""
    if "eigimg" in src.name:
        # name of `src` is of form "abide-eig/data/eigimgs/NYU_0051015_func_minimal_eigimg.nii.gz"
        # name of eigs file is of form "abide-eig/data/eigs/NYU_0051015_func_minimal.npy"
        eigs = np.load(EIGS / src.name.replace("_eigimg.nii.gz", ".npy"))
        return cast(ndarray, cropped / eigs)
    return cast(ndarray, cropped / np.mean(cropped, axis=(0, 1, 2)))


def save(args: PreprocArgs, img: ndarray) -> Path:
    src = args.nii
    norm = "_norm" if args.normalize else ""
    outdir: Path = DEEP_EIGIMG if "eigimg" in src.name else DEEP_FMRI
    outfile = outdir / src.name.replace(".nii.gz", f"_cropped{norm}.npy")
    np.save(outfile, img.astype(np.float32), allow_pickle=False)
    return outfile


def preprocess_image(args: PreprocArgs) -> None:
    nii = args.nii
    mask = args.mask
    cropper = args.cropper
    try:
        cropped = crop_to_bounds(nii, mask, cropper)
        result = normalize(nii, cropped) if args.normalize else cropped
        save(args, result)
    except Exception as e:
        print(f"Got exception {e} for image {nii}")
        traceback.print_exc()


def preprocess_images(is_eigimg: bool = False, transforms: List[Transform] = []) -> None:
    mask = nib.load(str(MASK)).get_fdata().astype(bool)
    cropper = get_mask_bounds(mask)
    src_dir = EIGIMGS if is_eigimg else NIIS

    niis = sorted(src_dir.rglob("*.nii.gz"))
    args = [
        PreprocArgs(nii=nii, mask=mask, cropper=cropper, normalize=True, transforms=transforms)
        for nii in niis
    ]
    process_map(preprocess_image, args)


if __name__ == "__main__":
    mask = nib.load(str(MASK)).get_fdata().astype(bool)
    cropper = get_mask_bounds(mask)
    for SRC_DIR in [EIGIMGS, NIIS]:
        print(f"Preprocessing images in {SRC_DIR}")
        niis = sorted(SRC_DIR.rglob("*.nii.gz"))
        args = [
            PreprocArgs(nii=nii, mask=mask, cropper=cropper, normalize=True, transforms=[])
            for nii in niis
        ]
        process_map(preprocess_image, args)
