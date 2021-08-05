# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on
import glob
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, cast, no_type_check

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import pytest
import torch
from numpy import ndarray
from pandas import DataFrame, Series
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from typing_extensions import Literal

from src.analysis.predict.reducers import subject_labels

DATA = ROOT / "data"
DEEP = DATA / "deep"  # for DL preprocessed fMRI data
DEEP_FMRI = DEEP / "fmri"
DEEP_EIGIMG = DEEP / "eigimg"
# Not all images convert to eigenimg of same dims, so we only use fMRI
# images that we could compute comparable eigimgs for.
PREPROC_EIG = sorted(DEEP_EIGIMG.rglob("*.npy"))
PREPROC_FMRI = [DEEP_FMRI / str(p.name).replace("_eigimg", "") for p in PREPROC_EIG]
print("Verifying fMRI files match eigimg files...")
n_exist = 0
for fmri in tqdm(PREPROC_FMRI, total=len(PREPROC_EIG), disable=True):
    assert fmri.exists(), f"Matching fMRI files: {n_exist}. Currently missing: {fmri}"
    n_exist += 1
LABELS: List[int] = subject_labels(PREPROC_EIG)
info = DataFrame({"img": map(lambda p: p.name, PREPROC_EIG), "label": LABELS}, index=list(range(len(LABELS))))
ctrl = info.loc[info["label"] == 0, :]
auts = info.loc[info["label"] == 1, :]
ctrl = ctrl.iloc[:50, :]
auts = auts.iloc[:50, :]
df = pd.concat([ctrl, auts], axis=0)
df.to_csv(DEEP / "subjs.csv")
sys.exit()


# nii paths
IMGS: List[Path] = sorted(Path(__file__).resolve().parent.rglob("*nii.gz"))
# Path to a custom csv file with the file name, subject id, and diagnosis
ANNOTATIONS: DataFrame = pd.read_json(Path(__file__).resolve().parent / "subj_data.json")
FmriSlice = Tuple[int, int, int, int]  # just a convencience type to save space


class RandomFmriPatchDataset(Dataset):
    """Just grabs a random patch of size `patch_shape` from a random brain.

    Parameters
    ----------
    patch_shape: Tuple[int, int, int, int]
        The patch size.

    standardize: bool = True
        Whether or not to do intensity normalization before returning the Tensor.

    transform: Optional[Callable] = None
        The transform to apply to the 4D array.

    Notes
    -----
    This random loader is much simpler to implement and understand, but is a bit unusual. You may
    find it helpful to read through.
    """

    def __init__(
        self,
        patch_shape: Optional[FmriSlice] = None,
        standardize: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.annotations = ANNOTATIONS
        self.img_paths = IMGS
        self.labels: List[int] = subject_labels(self.img_paths)
        self.shapes: List[Tuple[int, int, int, int]] = []
        for img in IMGS:  # get the diagnosis, 0 = control, 1 = autism and other info
            file_id = img.stem.replace("_func_minimal.nii", "")
            label_idx = self.annotations["FILE_ID"] == file_id
            self.labels.append(int(self.annotations.loc[label_idx, "DX_GROUP"]))
            self.shapes.append(nib.load(str(img)).shape)  # usually (61, 73, 61, 236)
        self.max_dims = np.max(self.shapes, axis=0)
        self.min_dims = np.min(self.shapes, axis=0)

        self.standardize = standardize
        self.transform = transform

        # ensure patch shape is valid
        if patch_shape is None:
            smallest_dims = np.min(self.shapes, axis=0)[:-1]  # exclude time dim
            self.patch_shape = (*smallest_dims, 8)
        else:
            if len(patch_shape) != 4:
                raise ValueError("Patches must be 4D for fMRI")
            for dim, max_dim in zip(patch_shape, self.max_dims):
                if dim > max_dim:
                    raise ValueError("Patch size too large for data")
            self.patch_shape = patch_shape

    def __len__(self) -> int:
        # when generating the random dataloader, the "length" is kind of phoney. You could make the
        # length be anything, e.g. 1000, 4962, or whatever. However, what you set as the length will
        # define the epoch size. Here, this amounts to 5 given I have 5 brains downloaded, but you
        # could choose a larger number.
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tensor:
        # just return a random patch
        path = np.random.choice(self.img_paths)
        img = nib.load(str(path))
        # going larger than max_idx will put us past the end of the array
        max_idx = np.array(img.shape) - np.array(self.patch_shape) + 1

        # Python has a `slice` object which you can use to index into things with the `[]` operator
        # we are going to build the slices we need to index appropriately into our niis with the
        # `.dataobj` trick
        slices = []
        for length, maximum in zip(self.patch_shape, max_idx):
            start = np.random.randint(0, maximum)
            slices.append(slice(start, start + length))
        array = img.dataobj[slices[0], slices[1], slices[2], slices[3]]

        if self.standardize:
            array -= np.mean(array)
            array /= np.std(array, ddof=1)
        return torch.Tensor(array)

    def test_get_item(self) -> None:
        """Just test that the produced slices can't ever go past the end of a brain"""
        for path in self.img_paths:
            img = nib.load(str(path))
            max_idx = np.array(img.shape) - np.array(self.patch_shape) + 1
            max_dims = img.shape
            for length, maximum, max_dim in zip(self.patch_shape, max_idx, max_dims):
                for start in range(maximum):
                    # array[a:maximum] is to the end
                    assert start + length <= max_dim
                    if start == maximum - 1:
                        assert start + length == max_dim


class FmriPatchDataset(Dataset):
    """Creates a proper map-style dataset (https://pytorch.org/docs/stable/data.html#map-style-datasets)

    Parameters
    ----------
    patch_shape: Tuple[int, int, int, int]
        The patch size.

    strides: Tuple[int, int, int, int] = (1, 1, 1, 1)
        How far to slide each patch from the previous.

    standardize: bool = True
        Whether or not to do intensity normalization before returning the Tensor.

    transform: Optional[Callable] = None
        The transform to apply to the 4D array.
    """

    def __init__(
        self,
        patch_shape: Optional[Tuple[int, int, int, int]] = None,
        strides: Tuple[int, int, int, int] = (1, 1, 1, 1),
        standardize: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.annotations = ANNOTATIONS
        self.img_paths = IMGS
        self.labels: List[int] = []
        self.shapes: List[Tuple[int, int, int, int]] = []
        self.strides = strides
        for img in IMGS:  # get the diagnosis, 0 = control, 1 = autism and other info
            file_id = img.stem.replace("_func_minimal.nii", "")
            label_idx = self.annotations["FILE_ID"] == file_id
            self.labels.append(int(self.annotations.loc[label_idx, "DX_GROUP"]))
            self.shapes.append(nib.load(str(img)).shape)  # usually (61, 73, 61, 236)
        self.max_dims = np.max(self.shapes, axis=0)
        self.min_dims = np.min(self.shapes, axis=0)

        self.standardize = standardize
        self.transform = transform

        # ensure patch shape is valid
        if patch_shape is None:
            smallest_dims = np.min(self.shapes, axis=0)[:-1]  # exclude time dim
            self.patch_shape = (*smallest_dims, 8)
        else:
            if len(patch_shape) != 4:
                raise ValueError("Patches must be 4D for fMRI")
            for dim, max_dim in zip(patch_shape, self.max_dims):
                if dim > max_dim:
                    raise ValueError("Patch size too large for data")
            self.patch_shape = patch_shape

        # now we need to know how many patches are produced for the dataset given a valid data size
        # for a 1D array, if you have a window size of `w`, and a length of `n`, there are
        # `n - w + 1` windows given a stride of 1.
        #
        # now the problem is you have to make an index `i` give you one unique patch from one brain.
        # the easy (but slow) way to do this is to loop over all brains and all patches and form a
        # dict of indices to the patches and other info needed:
        self.patches_dict: Dict[
            int, Tuple[Path, FmriSlice, int]
        ] = {}  # dict of slice starts, paths, labels
        count = 0
        (H, W, D, T) = self.patch_shape
        (H_s, W_s, D_s, T_s) = self.strides
        for path, shape, label in tqdm(
            zip(self.img_paths, self.shapes, self.labels), total=len(self.shapes)
        ):
            for m in range(0, shape[3] - T + 1, T_s):
                for k in range(0, shape[2] - D + 1, D_s):
                    for j in range(0, shape[1] - W + 1, W_s):
                        for i in range(0, shape[0] - H + 1, H_s):
                            self.patches_dict[count] = (path, (i, j, k, m), label)
                            count += 1
        self.n_patches = count

    def __len__(self) -> int:
        return self.n_patches

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        path, starts, label = self.patches_dict[index]
        img = nib.load(str(path))

        # build up the slices as above for indexing into `.dataobj`
        slices = []
        for i, start in enumerate(starts):
            slices.append(slice(start, start + self.patch_shape[i]))
        array = img.dataobj[slices[0], slices[1], slices[2], slices[3]]

        if self.standardize:
            array -= np.mean(array)
            array /= np.std(array, ddof=1)
        x = torch.Tensor(array)
        y = label
        if self.transform is None:
            return x, y
        return self.transform(x), y

    def test_get_item(self) -> None:
        """Just test that the produced slices can't ever go past the end of a brain"""
        for index in tqdm(range(self.n_patches), total=self.n_patches):
            path, starts, _ = self.patches_dict[index]
            img = nib.load(str(path))
            for i, start in enumerate(starts):
                assert start + self.patch_shape[i] <= img.shape[i]


def plot_patch(size: Tuple[int, int, int, int] = (32, 32, 32, 12)) -> None:
    dl = RandomFmriPatchDataset(size)
    img = dl[0].numpy()
    img -= np.mean(img)
    img /= np.std(img, ddof=1)
    img = np.clip(img, -5, 5)
    fig, axes = plt.subplots(nrows=1, ncols=3)
    axes.flat[0].imshow(img[:, :, 0, 0], cmap="Greys")
    axes.flat[1].imshow(img[:, 0, :, 0], cmap="Greys")
    axes.flat[2].imshow(img[0, :, :, 0], cmap="Greys")
    plt.show()


class FmriDataset(Dataset):
    """Loads entire 4D images.

    Parameters
    ----------
    standardize: bool = True
        Whether or not to do intensity normalization before returning the Tensor.

    transform: Optional[Callable] = None
        The transform to apply to the 4D array.

    is_eigimg: bool = False
        If False (default) loads the registered and minimially pre-processed ABIDE fMRI scans where
        controls are truncated to 175 timepoints. If True, loads the computed eigenimages with 175
        timepoints.

    Notes
    -----
    This random loader is much simpler to implement and understand, but is a bit unusual. You may
    find it helpful to read through.
    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        is_eigimg: bool = False,
    ) -> None:
        self.mask = nib.load(MASK).get_fdata().astype(bool)  # more efficient to load just once
        self.img_paths = EIGIMGS if is_eigimg else NIIS
        self.labels: List[int] = []
        self.shapes: List[Tuple[int, int, int, int]] = []
        for img in IMGS:  # get the diagnosis, 0 = control, 1 = autism and other info
            file_id = img.stem.replace("_func_minimal.nii", "")
            label_idx = self.annotations["FILE_ID"] == file_id
            self.labels.append(int(self.annotations.loc[label_idx, "DX_GROUP"]))
            self.shapes.append(nib.load(str(img)).shape)  # usually (61, 73, 61, 236)
        self.max_dims = np.max(self.shapes, axis=0)
        self.min_dims = np.min(self.shapes, axis=0)

        self.standardize = standardize
        self.transform = transform

        # ensure patch shape is valid
        if patch_shape is None:
            smallest_dims = np.min(self.shapes, axis=0)[:-1]  # exclude time dim
            self.patch_shape = (*smallest_dims, 8)
        else:
            if len(patch_shape) != 4:
                raise ValueError("Patches must be 4D for fMRI")
            for dim, max_dim in zip(patch_shape, self.max_dims):
                if dim > max_dim:
                    raise ValueError("Patch size too large for data")
            self.patch_shape = patch_shape

    def __len__(self) -> int:
        # when generating the random dataloader, the "length" is kind of phoney. You could make the
        # length be anything, e.g. 1000, 4962, or whatever. However, what you set as the length will
        # define the epoch size. Here, this amounts to 5 given I have 5 brains downloaded, but you
        # could choose a larger number.
        return len(self.annotations)

    def __getitem__(self, index: int) -> Tensor:
        # just return a random patch
        path = np.random.choice(self.img_paths)
        img = nib.load(str(path))
        # going larger than max_idx will put us past the end of the array
        max_idx = np.array(img.shape) - np.array(self.patch_shape) + 1

        # Python has a `slice` object which you can use to index into things with the `[]` operator
        # we are going to build the slices we need to index appropriately into our niis with the
        # `.dataobj` trick
        slices = []
        for length, maximum in zip(self.patch_shape, max_idx):
            start = np.random.randint(0, maximum)
            slices.append(slice(start, start + length))
        array = img.dataobj[slices[0], slices[1], slices[2], slices[3]]

        if self.standardize:
            array -= np.mean(array)
            array /= np.std(array, ddof=1)
        return torch.Tensor(array)

    def test_get_item(self) -> None:
        """Just test that the produced slices can't ever go past the end of a brain"""
        for path in self.img_paths:
            img = nib.load(str(path))
            max_idx = np.array(img.shape) - np.array(self.patch_shape) + 1
            max_dims = img.shape
            for length, maximum, max_dim in zip(self.patch_shape, max_idx, max_dims):
                for start in range(maximum):
                    # array[a:maximum] is to the end
                    assert start + length <= max_dim
                    if start == maximum - 1:
                        assert start + length == max_dim
