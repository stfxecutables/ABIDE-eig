# fmt: off
from concurrent.futures import process

import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
import torch
from pandas import DataFrame
from torch import Tensor
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.analysis.predict.reducers import subject_labels

DATA = ROOT / "data"
DEEP = DATA / "deep"  # for DL preprocessed fMRI data
DEEP_FMRI = DEEP / "fmri"
DEEP_EIGIMG = DEEP / "eigimg"
# Not all images convert to eigenimg of same dims, so we only use fMRI
# images that we could compute comparable eigimgs for.
PREPROC_EIG = sorted(DEEP_EIGIMG.rglob("*.npy"))
PREPROC_FMRI = [DEEP_FMRI / str(p.name).replace("_eigimg", "") for p in PREPROC_EIG]
LABELS: List[int] = subject_labels(PREPROC_EIG)
SHAPE = (47, 59, 42, 175)


def verify(fmri_eig: Tuple[Path, Path]) -> bool:
    fmri, eigimg = fmri_eig
    fail = False
    if not fmri.exists():
        print(f"Matching fMRI files currently missing: {fmri}")
        fail = True
    if not eigimg.exists():
        print(f"Matching fMRI files currently missing eigimg: {eigimg}")
        fail = True
    if not np.load(fmri).shape == SHAPE:
        print(f"Invalid input shape for file {fmri}")
        fail = True
    if not np.load(eigimg).shape == SHAPE:
        print(f"Invalid input shape for file {eigimg}")
        fail = True
    return fail


def verify_matching() -> None:
    print("Verifying fMRI files match eigimg files... ", end="", flush=True)
    n_unmatched = np.sum(
        process_map(
            verify,
            list(zip(PREPROC_FMRI, PREPROC_EIG)),
            total=len(PREPROC_EIG),
            desc="Verifying fMRI/eig matches",
        )
    )
    if n_unmatched > 0:
        print(f"Failure to verify. {n_unmatched} unmatched files.")


def get_testing_subsample() -> None:
    info = DataFrame(
        {"img": map(lambda p: p.name, PREPROC_EIG), "label": LABELS}, index=list(range(len(LABELS)))
    )
    ctrl = info.loc[info["label"] == 0, :]
    auts = info.loc[info["label"] == 1, :]
    ctrl = ctrl.iloc[:50, :]
    auts = auts.iloc[:50, :]
    df = pd.concat([ctrl, auts], axis=0)
    df.to_csv(DEEP / "subjs.csv")
    sys.exit()


# get_testing_subsample()
# verify_matching()

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
    Only intended for quick testing.
    """

    def __init__(
        self,
        patch_shape: FmriSlice = SHAPE,
        standardize: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.annotations = LABELS
        self.img_paths = PREPROC_EIG
        self.labels: List[int] = subject_labels(self.img_paths)

        self.standardize = standardize
        self.transform = transform

        # ensure patch shape is valid
        if len(patch_shape) != 4:
            raise ValueError("Patches must be 4D for fMRI")
        for i, (dim, max_dim) in enumerate(zip(patch_shape, SHAPE)):
            if dim > max_dim:
                raise ValueError(
                    f"Patch size in dim {i} ({dim}) too large for data, "
                    f"which has at dim {i} size {max_dim}"
                )
        self.patch_shape = patch_shape

    def __len__(self) -> int:
        return len(LABELS)

    def __getitem__(self, index: int) -> Tensor:
        img = np.load(np.random.choice(self.img_paths))
        # going larger than max_idx will put us past the end of the array
        max_idx = np.array(img.shape) - np.array(self.patch_shape) + 1

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
    """Generate patches.

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
        patch_shape: Tuple[int, int, int, int] = SHAPE,
        strides: Tuple[int, int, int, int] = (1, 1, 1, 1),
        standardize: bool = True,
        transform: Optional[Callable] = None,
    ) -> None:
        self.annotations = LABELS
        self.img_paths = PREPROC_EIG
        self.labels: List[int] = subject_labels(self.img_paths)
        self.strides = strides

        self.standardize = standardize
        self.transform = transform

        # ensure patch shape is valid
        if len(patch_shape) != 4:
            raise ValueError("Patches must be 4D for fMRI")
        for i, (dim, max_dim) in enumerate(zip(patch_shape, SHAPE)):
            if dim > max_dim:
                raise ValueError(
                    f"Patch size in dim {i} ({dim}) too large for data, "
                    f"which has at dim {i} size {max_dim}"
                )
        self.patch_shape = patch_shape

        # now we need to know how many patches are produced for the dataset given a valid data size
        # for a 1D array, if you have a window size of `w`, and a length of `n`, there are
        # `n - w + 1` windows given a stride of 1. Now just make an index `i` give you one unique
        # patch from one brain
        self.patches_dict: Dict[
            int, Tuple[Path, FmriSlice, int]
        ] = {}  # dict of slice starts, paths, labels
        count = 0
        (H, W, D, T) = self.patch_shape
        (H_s, W_s, D_s, T_s) = self.strides
        for path, label in tqdm(zip(self.img_paths, self.labels), total=len(self.shapes)):
            for m in range(0, SHAPE[3] - T + 1, T_s):
                for k in range(0, SHAPE[2] - D + 1, D_s):
                    for j in range(0, SHAPE[1] - W + 1, W_s):
                        for i in range(0, SHAPE[0] - H + 1, H_s):
                            self.patches_dict[count] = (path, (i, j, k, m), label)
                            count += 1
        self.n_patches = count

    def __len__(self) -> int:
        return self.n_patches

    def __getitem__(self, index: int) -> Tuple[Tensor, int]:
        path, starts, label = self.patches_dict[index]
        img = np.load(path)

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
            img = np.load(path)
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


def preload(img: Path) -> np.ndarray:
    # need channels_first, but is currently channels last
    x = np.load(img)
    x = np.transpose(x, (3, 0, 1, 2))
    x -= np.mean(x)
    x /= np.std(x, ddof=1)
    return x


class FmriDataset(Dataset):
    """Loads entire 4D images.

    Parameters
    ----------
    transform: Optional[Callable] = None
        The transform to apply to the 4D array.

    is_eigimg: bool = False
        If False (default) loads the registered and minimially pre-processed ABIDE fMRI scans where
        controls are truncated to 175 timepoints. If True, loads the computed eigenimages with 175
        timepoints.

    preload: bool = False
        If True, load all images into memory (over 100 GB in full case)
    """

    def __init__(
        self,
        transform: Optional[Callable] = None,
        is_eigimg: bool = False,
        preload_data: bool = False,
    ) -> None:
        # self.mask = nib.load(MASK).get_fdata().astype(bool)  # more efficient to load just once
        self.preload = bool(preload_data)
        self.img_paths = PREPROC_EIG if is_eigimg else PREPROC_FMRI
        self.transform = transform
        self.imgs = []
        if self.preload:
            self.imgs = process_map(
                preload, self.img_paths, total=len(self.img_paths), desc="Preloading all images..."
            )

    def __len__(self) -> int:
        return len(self.img_paths)

    def __getitem__(self, i: int) -> Tuple[Tensor, Tensor]:
        y = int(LABELS[i])
        if self.preload:
            x = self.imgs[i]
            return Tensor(x), Tensor([y])
        # need channels_first, but is currently channels last
        x = np.load(self.img_paths[i])
        x = np.transpose(x, (3, 0, 1, 2))
        x -= np.mean(x)
        x /= np.std(x, ddof=1)
        return Tensor(x), Tensor([y])
