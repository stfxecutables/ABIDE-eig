import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import traceback
from multiprocessing import Pool, RawArray  # type: ignore
from pathlib import Path
from pickle import UnpicklingError, dump, load
from time import time
from typing import Any, Dict, List, Tuple, Union, cast

import nibabel as nib
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm
from typing_extensions import Literal

from src.constants import CKPT_PATH

MASK = Path(__file__).resolve().parent.parent.parent / "data/atlases/MASK.nii.gz"
SHM_FLAT_NAME = "flat_nii_array"
SHM_MASK_NAME = "mask_nii_array"
GLOBALS = {}
DTYPE = np.float32
N_PROCESSES = 84 if os.environ.get("CC_CLUSTER") == "niagara" else 8
OPTIMAL_CHUNKSIZE = 1
N_VOXELS_TIME_ESTIMATION = 3000


def eigs_via_transpose(M: ndarray, covariance: bool = True) -> ndarray:
    """Use transposes to rapidly compute eigenvalues of covariance and
    correlation matrices.

    Parameters
    ----------
    M: ndarray
        A time-series array with shape (N, T) such that N > T. (If N <= T, no
        benefits are gained from transposed intermediates, so the eigenvalues
        are simply computed in the normal fashion).

    Returns
    -------
    eigs: ndarray
        The computed eigenvalues.

    Notes
    -----
    # Covariance Matrices

        if:
            (n, p) = X.shape, n > p, AND
            norm(X) = X - np.mean(X, axis=1, keepdims=True),
            Z = norm(X)
        then:
            np.cov(X)  == np.matmul(norm(X), norm(X).T) / (p - 1)
                       == r * Z * Z.T,   if   r = 1/(1-p)

        Now, it then follows that:

            eigs(np.cov(X))  ==  eigs(r Z * Z.T)
                             == r * eigs(Z * Z.T)

        But the nonzero eigenvalues of Z*Z.T are exactly those of Z.T*Z. Thus:
            eigs_nz(np.cov(X))  == r * eigs(Z.T * Z)
        which can be computed extremely quickly if p << n

    # Correlation Matrices

        Keeping X, n, p, r the same as above, and letting C = corr(X), and

            sd = np.std(X, axis=1, ddof=1, keepdims=True)
            m = np.mean(X, axis=1, keepdims=True)
            stand(X) == (X - m) / sd == norm(X) /sd

        we have that corr(X) == np.cov(stand(X)), but then since

            norm(stand(X)) = stand(X),

        if we let Y = stand(X), Z = norm(Y), then

            eigs(corr(X)) == eigs(np.cov(stand(X)))
                          == eigs(np.cov(Y))

        Now, as above, it then follows that:

            eigs_nz(corr(X)) == r * eigs(Z.T*Z)
                             == r * eigs(Y.T*Y)

    """
    N, T = M.shape
    if N <= T:
        raise ValueError("Array is not of correct shape to benefit from transposed intermediates.")
    r = 1 / (T - 1)
    Z = M - np.mean(M, axis=1, keepdims=True)
    if not covariance:
        Z /= np.std(M, axis=1, ddof=1, keepdims=True)
    M = np.matmul(Z.T, r * Z)
    eigs: ndarray = np.linalg.eigvalsh(M)
    return eigs


def full_eigensignal(nii: Path, mask: Path, covariance: bool = True) -> ndarray:
    """Get the full eigenvalues over all brain tissue"""
    array4d = nib.load(str(nii)).get_fdata()
    msk = nib.load(str(mask)).get_data().astype(bool)
    brain = array4d[msk, :]
    eigs: ndarray = eigs_via_transpose(brain, covariance=covariance)
    return eigs[1:]  # type: ignore


# will take about 8 hours per image on PARK dataset, using all 8 cores for
# eigenvalue calculation
# DEPRECATE
def compute_eigencontribution_slow(
    array4d: ndarray, covariance: bool = True, progress_bar=True
) -> ndarray:
    N, t = int(np.prod(array4d.shape[:-1])), int(array4d.shape[-1])
    eig_length = t - 1
    end_shape = array4d.shape[:-1] + (array4d.shape[-1] - 1,)

    flat = np.reshape(array4d, [N, t])  # for looping
    mask = (array4d.sum(axis=-1) != 0) & (array4d.std(axis=-1) != 0)
    maskflat = mask.ravel()
    n_voxels = np.sum(maskflat)

    # we need to loop over the flat array, but be 100% sure that we can unflatten the
    # array while preserving the original shape
    contrib = np.full(shape=[N, eig_length], fill_value=np.nan, dtype=float)
    times = []
    signal_voxels = 0
    converge_fails = 0
    for voxel in tqdm(range(flat.shape[0]), disable=not progress_bar):
        if maskflat[voxel] == 0:
            contrib[voxel, 0] = -1  # flag value, impossible because pos. semi-definite
            continue
        signal_voxels += 1
        start = time()
        voxelmask = np.arange(flat.shape[0]) != voxel
        deleted = flat[voxelmask, :]
        try:
            eigs = eigs_via_transpose(deleted, covariance=covariance)
        except:  # in case it doesn't conververge # noqa
            try:
                eigs = eigs_via_transpose(deleted, covariance=covariance)
            except:  # noqa
                converge_fails += 1
                traceback.print_exc()
                eigs = np.zeros([end_shape[-1]], dtype=float) - 1

        contrib[voxel] = eigs[1:]  # delete smallest (zero) eigenvalue
        times.append(time() - start)
        if len(times) % 10 == 0:
            avg = np.mean(times)
            print(
                f"Average time per voxel: {np.mean(times)}s. "
                f"Expected total time remaining: {avg * (n_voxels-signal_voxels) / 60 / 60} hrs."
                f"Convergence fails so far: {converge_fails}."
            )

    return contrib.reshape(end_shape)


def compute_eigensignal(argsdict: Dict) -> ndarray:
    flat = argsdict["flat"]
    maskflat = argsdict["maskflat"]
    voxel = argsdict["voxel"]
    covariance = argsdict["covariance"]
    T = flat.shape[1] - 1
    if maskflat[voxel] == 0:
        return np.full([T], np.nan, dtype=float)
    voxelmask = np.arange(flat.shape[0]) != voxel
    deleted = flat[voxelmask, :]
    try:
        eigs = eigs_via_transpose(deleted, covariance=covariance)
        return eigs[1:]  # type: ignore
    except:  # in case it doesn't conververge  # noqa
        try:
            eigs = eigs_via_transpose(deleted, covariance=covariance)
            return eigs[1:]  # type: ignore
        except:  # noqa
            traceback.print_exc()
            eigs = np.full([T], -1, dtype=float)
            return eigs


def init(
    flat_ptr: RawArray,
    flat_shape: Tuple[int, ...],
    mask_ptr: RawArray,
    mask_shape: Tuple[int, ...],
) -> None:
    global GLOBALS
    GLOBALS["flat_ptr"] = flat_ptr
    GLOBALS["flat_shape"] = flat_shape
    GLOBALS["mask_ptr"] = mask_ptr
    GLOBALS["mask_shape"] = mask_shape


# https://gist.github.com/rossant/7a46c18601a2577ac527f958dd4e452f
# def mem_ptr_and_numpy_view(dtype: np.dtype, shape: Tuple[int, ...]) -> Tuple[RawArray, ndarray]:
#     dtype = np.dtype(dtype)
#     # Get a ctype type from the NumPy dtype.
#     ctype = np.ctypeslib.as_ctypes_type(dtype)
#     # Create the RawArray instance.
#     mem_ptr = RawArray(ctype, sum(shape))
#     # Get a NumPy array view.
#     view = np.frombuffer(mem_ptr, dtype=dtype).reshape(shape)
#     return mem_ptr, view


def mem_ptr_and_numpy_view(array: ndarray) -> Tuple[RawArray, ndarray]:
    dtype = np.dtype(DTYPE)
    ctype = np.ctypeslib.as_ctypes_type(dtype)  # Get a ctype type from the NumPy dtype.
    mem_ptr = RawArray(ctype, int(np.prod(array.shape)))
    view = np.frombuffer(mem_ptr, dtype=DTYPE).reshape(array.shape)
    view[:] = array[:]
    return mem_ptr, view


def eigsignal_from_shared(argsdict: Dict) -> ndarray:
    flat = np.frombuffer(GLOBALS["flat_ptr"], dtype=DTYPE).reshape(GLOBALS["flat_shape"])
    maskflat = np.frombuffer(GLOBALS["mask_ptr"], dtype=DTYPE).reshape(GLOBALS["mask_shape"])

    voxel = argsdict["voxel"]
    covariance = argsdict["covariance"]
    T = flat.shape[1] - 1
    if maskflat[voxel] == 0:
        return np.full([T], np.nan, dtype=DTYPE)
    voxelmask = np.arange(flat.shape[0]) != voxel
    deleted = flat[voxelmask, :]
    try:
        eigs = eigs_via_transpose(deleted, covariance=covariance)
        return eigs[1:]  # type: ignore
    except Exception as e:  # noqa
        traceback.print_exc()
        print(e)
        try:
            eigs = eigs_via_transpose(deleted, covariance=covariance)
            return eigs[1:]  # type: ignore
        except Exception as e:  # noqa
            traceback.print_exc()
            print(e)
            return np.full([T], -1, dtype=DTYPE)


def parallel_setup(
    array4d: ndarray, flat: ndarray, covariance: bool, estimate_time: bool, decimation: int
) -> Tuple[ndarray, List[Dict[str, Any]]]:
    # we need to loop over the flat array, but be 100% sure that we can unflatten the
    # array while preserving the original shape
    N, t = int(np.prod(array4d.shape[:-1])), int(array4d.shape[-1])
    eig_length = t - 1
    contrib = np.full(shape=[N, eig_length], fill_value=np.nan, dtype=float)
    step = decimation if estimate_time else 1  # decimate for testing
    args = [
        dict(
            voxel=voxel,
            covariance=covariance,
        )
        for voxel in range(0, flat.shape[0], step)  # decimation means ~4800 voxels
    ]
    return contrib, args


def shared_setup(array4d: ndarray) -> Tuple[ndarray, ndarray, RawArray, ndarray, RawArray, ndarray]:
    N, t = int(np.prod(array4d.shape[:-1])), int(array4d.shape[-1])
    flat = np.reshape(array4d, [N, t]).astype(DTYPE)  # for looping
    # mask = (array4d.sum(axis=-1) != 0) & (array4d.std(axis=-1) != 0)
    mask = nib.load(str(MASK)).get_data().astype(bool)
    maskflat = mask.ravel().astype(DTYPE)
    # parallel setup
    flat_ptr, flat_view = mem_ptr_and_numpy_view(flat)
    mask_ptr, mask_view = mem_ptr_and_numpy_view(maskflat)
    return maskflat.shape, flat, flat_ptr, flat_view, mask_ptr, mask_view


def find_optimal_chunksize(
    array4d: ndarray,
    covariance: bool = True,
    n_voxels: int = 3000,
) -> DataFrame:
    mask_shape, flat, flat_ptr, flat_view, mask_ptr, mask_view = shared_setup(array4d)

    # we need to loop over the flat array, but be 100% sure that we can unflatten the
    # array while preserving the original shape
    CHUNKSIZES = [1]
    df = DataFrame(columns=["Chunksize", "Duration (s)"])
    print(f"Using {N_PROCESSES} processes.")
    row = 0
    for chunksize in CHUNKSIZES:
        args = [
            dict(
                voxel=voxel,
                covariance=covariance,
            )
            for voxel in range(flat.shape[0])
        ][:n_voxels]
        print(f"Beginning analysis for chunksize={chunksize}")
        start = time()
        with Pool(
            processes=N_PROCESSES,
            initializer=init,
            initargs=(flat_view, flat.shape, mask_view, mask_shape),
        ) as pool:
            _ = pool.map(eigsignal_from_shared, args, chunksize=chunksize)
        duration = time() - start
        df.loc[row, :] = (chunksize, duration)
        row += 1
        print(df)
    print(df)
    return df


def estimate_computation_time(
    nii: Path,
    covariance: bool = True,
    estimate_time: bool = False,
    decimation: int = 64,
) -> pd.Timedelta:
    # loading setup
    array4d = nib.load(str(nii)).get_fdata()
    T = int(array4d.shape[-1])
    mask_shape, flat, flat_ptr, flat_view, mask_ptr, mask_view = shared_setup(array4d)
    contrib, args = parallel_setup(array4d, flat, covariance, estimate_time, decimation)

    print(f"Estimating time for subject with T = {T} using {len(args)} voxels. ", end="")
    print(f"Using {N_PROCESSES} processes.")
    start = time()

    with Pool(
        processes=N_PROCESSES,
        initializer=init,
        initargs=(flat_view, flat.shape, mask_view, mask_shape),
    ) as pool:
        signals = pool.map(eigsignal_from_shared, args, chunksize=OPTIMAL_CHUNKSIZE)

    elapsed = time() - start  # seconds
    total_voxels = np.prod(array4d.shape[:-1])
    computed_voxels = len(args)
    total_seconds = (elapsed / computed_voxels) * total_voxels
    duration = pd.Timedelta(seconds=total_seconds)
    return duration


def ckpt_paths(nii: Path) -> Tuple[Path, Path]:
    # re-suffix
    p = Path(nii.name)
    extensions = "".join(p.suffixes)
    ckpt_name = str(p).replace(extensions, ".ckpt")
    new_name = str(p).replace(extensions, ".ckpt.new")
    ckpt_path: Path = CKPT_PATH / ckpt_name
    new_path: Path = CKPT_PATH / new_name
    return ckpt_path, new_path


def is_corrupt(path: Path) -> bool:
    try:
        with open(path, "rb") as file:
            load(file)
        return True
    except (UnpicklingError, EOFError):
        return False
    except BaseException as e:
        traceback.print_exc()
        print(e)
        return False


def cleanup(nii: Path) -> None:
    ckpt, new = ckpt_paths(nii)
    if not new.exists():  # nothing to do
        return
    if is_corrupt(new):
        new.unlink()
        return
    new.replace(ckpt)


def load_checkpoint(nii: Path) -> List[ndarray]:
    """Checks `nii` path for associated checkpoint file, and if present loads the stored
    signals.
    """
    ckpt = ckpt_paths(nii)[0]
    cleanup(nii)  # just in case
    if ckpt.exists():
        with open(ckpt, "rb") as file:
            signals: List[ndarray] = load(file)
        return signals
    else:
        return []


def save_checkpoint(signals: List[ndarray], nii: Path) -> None:
    """We have to be a bit careful here because if we get interrupted during writing we overwrite
    our checkpoint. There are N locations where a harmful interruption can happen:

    1. writing `.ckpt.new` (slow)
    2. renaming `.ckpt.new` to `.ckpt` (fast)

    Fail at 1: (possible)
        There is a `.new` file. We need to check if the `.new` file is a valid pickle. If so, we
        need to rename it to `.ckpt`, otherwise, delete it and the indicator and return.
    Fail at 2: (extremely unlikely)
        Not even sure this can happen with Slurm, definitely seems impossible given the grace
        period and speed of this op
    """
    ckpt, new = ckpt_paths(nii)
    with open(new, "wb") as out:
        dump(signals, out)
    new.replace(ckpt)


def compute_eigenimage(
    nii: Path,
    covariance: bool = True,
    t: int = -1,
) -> Union[ndarray, pd.Timedelta]:
    """
    Notes
    -----
    We want to checkpoint this, but to do this we have to split the parallelization a bit.
    We know each eigenimage takes between 1-2.5 hours to compute. Doing e.g. 8-20 images per job we
    will thus often expect the last image to be interrupted, and often for a few images at the end
    not to be reached. Just losing the entire 1-3 hours is not really acceptable, so we need to
    checkpoint every K amount of eigensignals processed. Ideally we want this to be about every 20
    minutes or so, so it runs not too often and we lose at most 20 minutes of compute time.

    Since the longest jobs are about 2.5 hours (maybe 3 hours if there is something really bad or
    strange about the data) this means doing about 12 checkpoints per image.
    """
    N_SPLITS = 12
    # loading setup
    array4d = nib.load(str(nii)).get_fdata()
    if t > 0:
        array4d = array4d[:, :, :, -t:]
    # array4d = array4d[::5, ::5, ::5, :]  # testing
    spatial, T = array4d.shape[:-1], int(array4d.shape[-1])
    end_shape = (*spatial, T - 1)
    mask_shape, flat, flat_ptr, flat_view, mask_ptr, mask_view = shared_setup(array4d)
    contrib, args = parallel_setup(array4d, flat, covariance, estimate_time=False, decimation=1)

    signals = load_checkpoint(nii)
    if len(signals) == len(args):
        print(f"Computation for {nii} already computed. Restoring from final checkpoint.")
        return np.array(signals).reshape(end_shape)

    if len(signals) != 0:  # we are resuming from checkpoint so have lots of time
        print(f"Found checkpoint for file {nii}. Resuming...")
        args = args[len(signals) :]
        with Pool(
            processes=N_PROCESSES,
            initializer=init,
            initargs=(flat_view, flat.shape, mask_view, mask_shape),
        ) as pool:
            signals.extend(pool.map(eigsignal_from_shared, args, chunksize=OPTIMAL_CHUNKSIZE))
        print(f"Saving final checkpoint for {nii}...", end="")
        save_checkpoint(signals, nii)
        print(" done.")
        return np.array(signals).reshape(end_shape)

    # fresh start
    N = int(np.ceil(len(args) / N_SPLITS))
    splits = [slice(i * N, (i + 1) * N) for i in range(N_SPLITS)]
    for k, split in enumerate(splits):
        with Pool(
            processes=N_PROCESSES,
            initializer=init,
            initargs=(flat_view, flat.shape, mask_view, mask_shape),
        ) as pool:
            signals.extend(
                pool.map(eigsignal_from_shared, args[split], chunksize=OPTIMAL_CHUNKSIZE)
            )
        if k != len(splits) - 1:
            print(f"Saving checkpoint {k} for {nii}...", end="")
        else:
            print(f"Saving final checkpoint for {nii}...", end="")
        save_checkpoint(signals, nii)
        print(" done.")
    return np.array(signals).reshape(end_shape)
