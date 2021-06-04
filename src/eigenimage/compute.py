import multiprocessing as mp
from multiprocessing import RawArray, Pool
from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Sized, Tuple, Type, TypeVar, Union
from warnings import warn

import nibabel as nib
import os
import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal

SHM_FLAT_NAME = "flat_nii_array"
SHM_MASK_NAME = "mask_nii_array"
GLOBALS = {}
DTYPE = np.float64
N_PROCESSES = 44 if os.environ.get("CC_CLUSTER") == "niagara" else 8


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


def full_eigensignal(array4d: ndarray, covariance: bool = True) -> ndarray:
    """Get the full eigenvalues over all brain tissue"""
    mask = (array4d.sum(axis=-1) != 0) & (array4d.std(axis=-1) != 0)
    brain = array4d[mask, :]
    eigs: ndarray = eigs_via_transpose(brain, covariance=covariance)
    return eigs[1:]  # type: ignore


# will take about 8 hours per image on PARK dataset, using all 8 cores for
# eigenvalue calculation
def compute_eigencontribution(
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
        except:  # in case it doesn't conververge
            try:
                eigs = eigs_via_transpose(deleted, covariance=covariance)
            except:
                converge_fails += 1
                eigs = np.zeros([end_shape[-1]], dtype=float) - 1

        contrib[voxel] = eigs[1:]  # delete smallest (zero) eigenvalue
        times.append(time() - start)
        if len(times) % 10 == 0:
            avg = np.mean(times)
            print(
                f"Average time per voxel: {np.mean(times)}s. Expected total time remaining: {avg * (n_voxels-signal_voxels) / 60 / 60} hrs. Convergence fails so far: {converge_fails}."
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
        return eigs[1:]
    except:  # in case it doesn't conververge
        try:
            eigs = eigs_via_transpose(deleted, covariance=covariance)
            return eigs[1:]
        except:
            eigs = np.full([T], -1, dtype=float)
            return eigs

    return eigs[1:]


def eigimage_parallel(
    array4d: ndarray, covariance: bool = True, chunksize: int = 32, progress_bar: bool = True
):
    N, t = int(np.prod(array4d.shape[:-1])), int(array4d.shape[-1])
    eig_length = t - 1
    end_shape = array4d.shape[:-1] + (array4d.shape[-1] - 1,)

    flat = np.reshape(array4d, [N, t])  # for looping
    mask = (array4d.sum(axis=-1) != 0) & (array4d.std(axis=-1) != 0)
    maskflat = mask.ravel()

    # we need to loop over the flat array, but be 100% sure that we can unflatten the
    # array while preserving the original shape
    contrib = np.full(shape=[N, eig_length], fill_value=np.nan, dtype=float)
    args = [
        dict(flat=flat, maskflat=maskflat, voxel=voxel, covariance=covariance)
        for voxel in range(flat.shape[0])
    ]
    signals = process_map(compute_eigensignal, args, chunksize=chunksize, disable=not progress_bar)
    for i, signal in enumerate(signals):
        contrib[i] = signal
    return contrib.reshape(end_shape)


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
        return eigs[1:]
    except:  # in case it doesn't conververge
        try:
            eigs = eigs_via_transpose(deleted, covariance=covariance)
            return eigs[1:]
        except:
            eigs = np.full([T], -1, dtype=DTYPE)
            return eigs

    return eigs[1:]


def find_optimal_chunksize(
    array4d: ndarray, covariance: bool = True, progress_bar=True, n_voxels: int = 3000,
) -> DataFrame:
    # type setup
    dtype = np.float32

    # loading setup
    N, t = int(np.prod(array4d.shape[:-1])), int(array4d.shape[-1])
    eig_length = t - 1
    end_shape = array4d.shape[:-1] + (array4d.shape[-1] - 1,)
    flat = np.reshape(array4d, [N, t]).astype(DTYPE)  # for looping
    mask = (array4d.sum(axis=-1) != 0) & (array4d.std(axis=-1) != 0)
    maskflat = mask.ravel().astype(DTYPE)

    # parallel setup
    flat_shape = flat.shape
    mask_shape = maskflat.shape
    flat_ptr, flat_view = mem_ptr_and_numpy_view(flat)
    mask_ptr, mask_view = mem_ptr_and_numpy_view(maskflat)

    # shared_mem_flat = RawArray("d", int(np.prod(flat.shape)))
    # shared_flat = np.frombuffer(shared_mem_flat, dtype=np.float64).reshape(flat.shape)
    # shared_flat[:] = flat[:]

    # shared_mem_mask = RawArray("d", int(np.prod(maskflat.shape)))
    # shared_mask = np.frombuffer(shared_mem_mask, dtype=np.float64).reshape(maskflat.shape)
    # shared_mask[:] = maskflat[:]

    # we need to loop over the flat array, but be 100% sure that we can unflatten the
    # array while preserving the original shape
    contrib = np.full(shape=[N, eig_length], fill_value=np.nan, dtype=float)
    args = [
        dict(
            voxel=voxel,
            covariance=covariance,
        )
        for voxel in range(flat.shape[0])
    ][:n_voxels]
    CHUNKSIZES = [8, 16, 20, 24, 32, 40, 64, 128]
    df = DataFrame(index=pd.Index(CHUNKSIZES, name="chunksize"), columns=["Duration (s)"])
    print(f"Using {N_PROCESSES} processes.")
    for chunksize in CHUNKSIZES:
        print(f"Beginning analysis for chunksize={chunksize}")
        start = time()
        with Pool(
            processes=N_PROCESSES,
            initializer=init,
            initargs=(flat_view, flat.shape, mask_view, mask_shape),
        ) as pool:
            signals = pool.imap(eigsignal_from_shared, args, chunksize=chunksize)
            # signals = process_map(
            #     eigsignal_from_shared, args, chunksize=chunksize, disable=not progress_bar
            # )
        duration = time() - start
        df.loc[chunksize, "Duration (s)"] = duration
        print(df)
    print(df)
    return df
