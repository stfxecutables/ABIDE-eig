# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort:skip
sys.path.append(str(ROOT))  # isort:skip
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Type
from warnings import warn

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch
from pytorch_lightning import Trainer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torch.utils.data.dataset import Subset, random_split
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from src.analysis.predict.deep_learning.constants import INPUT_SHAPE
from src.analysis.predict.deep_learning.prepare_data import prepare_data_files
from src.analysis.predict.reducers import subject_labels

FmriSlice = Tuple[int, int, int, int]  # just a convencience type to save space

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


@dataclass
class PreloadArgs:
    img: Path
    slicer: slice


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


def preload(args: PreloadArgs) -> np.ndarray:
    # need channels_first, but is currently channels last
    img, slicer = args.img, args.slicer
    x = np.load(img)
    x = np.transpose(x, (3, 0, 1, 2))
    x -= np.mean(x)
    x /= np.std(x, ddof=1)
    return x[slicer]


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
        args: Namespace,
        transform: Optional[Callable] = None,
    ) -> None:
        # self.mask = nib.load(MASK).get_fdata().astype(bool)  # more efficient to load just once
        self.preload = bool(args.preload)
        self.img_paths = prepare_data_files(args.is_eigimg)
        self.transform = transform
        self.time_slice = args.slicer
        self.imgs = []
        if self.preload:
            pargs = [PreloadArgs(img=img, slicer=self.time_slice) for img in self.img_paths]
            self.imgs = process_map(
                preload, pargs, total=len(pargs), desc="Preloading all images..."
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
        x = x[self.time_slice]
        return Tensor(x), Tensor([y])

    def train_val_split(self, args: Namespace) -> Tuple[Subset, Subset]:
        test_length = 40 if len(self.img_paths) == 100 else 100
        train_length = len(self.img_paths) - test_length
        train, val = random_split(self, (train_length, test_length), generator=None)
        val_aut = torch.cat(list(zip(*list(val)))[1]).sum().int().item()  # type: ignore
        train_aut = torch.cat(list(zip(*list(train)))[1]).sum().int().item()  # type: ignore
        print("Subset sizes will be:")
        print(f"train: {len(train)} (Autism={train_aut}, Control={len(train) - train_aut})")
        print(f"val:   {len(val)} (Autism={val_aut}, Control={len(val) - val_aut})")

        if len(train) % args.batch_size != 0:
            warn(
                "Batch size does not evenly divide training set. "
                f"{len(train) % args.batch_size} subjects will be dropped each training epoch."
            )
        return train, val


def random_data() -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    # if our model is flexible enough we should be  able to overfit random data
    # we can!
    print("Generating class 1 training data")
    x1_train = torch.rand([20, *INPUT_SHAPE])
    print("Generating class 2 training data")
    x2_train = torch.rand([20, *INPUT_SHAPE])
    x_train = torch.cat([x1_train, x2_train])
    print("Normalizing")
    x_train -= torch.mean(x_train)
    y_train = torch.cat([torch.zeros(20), torch.ones(20)])
    print("Generating class 1 validation data")
    x1_val = torch.rand([5, *INPUT_SHAPE])
    print("Generating class 2 validation data")
    x2_val = torch.rand([5, *INPUT_SHAPE])
    x_val = torch.cat([x1_val, x2_val])
    print("Normalizing")
    x_val -= torch.mean(x_val)
    y_val = torch.cat([torch.zeros(5), torch.ones(5)])
    return x_train, y_train, x_val, y_val


class RandomSeparated(Dataset):
    def __init__(self, size: int) -> None:
        super().__init__()
        self.size = size

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor]:
        if index < self.size // 2:
            return torch.rand(INPUT_SHAPE) - 0.5, Tensor([0])
        return torch.rand(INPUT_SHAPE), Tensor([1])

    def __len__(self) -> int:
        return self.size


def test_overfit_random(
    model_class: Type, model_args: Dict, train_batch: int = 8, val_batch: int = 8
) -> None:
    X_TRAIN, Y_TRAIN, X_VAL, Y_VAL = random_data()
    train_loader = DataLoader(
        TensorDataset(X_TRAIN, Y_TRAIN),
        batch_size=train_batch,
        num_workers=8,
        shuffle=True,
        drop_last=True,
    )
    val_loader = DataLoader(TensorDataset(X_VAL, Y_VAL), batch_size=val_batch, num_workers=8)
    model = model_class(config=model_args)
    parser = ArgumentParser()
    parser.add_argument("--batch_size", default=train_batch, type=int)
    parser = Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.default_root_dir = ROOT / f"results/{model_class.__name__}_rand_test"
    trainer = Trainer.from_argparse_args(args)
    trainer.fit(model, train_loader, val_loader)
