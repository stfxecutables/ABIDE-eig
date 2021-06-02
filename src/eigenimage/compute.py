from copy import deepcopy
from pathlib import Path
from time import time
from typing import Any, Dict, List, Optional, Sized, Tuple, Type, TypeVar, Union
from warnings import warn

import nibabel as nib
import numpy as np
import pandas as pd
import scipy.sparse as sparse
from numpy import ndarray
from pandas import DataFrame
from scipy.integrate import quad
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map
from typing_extensions import Literal


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


def find_optimal_chunksize(
    array4d: ndarray, covariance: bool = True, progress_bar=True
) -> DataFrame:
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
    ][:400]
    CHUNKSIZES = [24, 32, 40, 48, 56]
    df = DataFrame(index=pd.Index(CHUNKSIZES, name="chunksize"), columns=["Duration (s)"])
    for chunksize in CHUNKSIZES:
        print(f"Beginning analysis for chunksize={chunksize}")
        start = time()
        signals = process_map(
            compute_eigensignal, args, chunksize=chunksize, disable=not progress_bar
        )
        duration = time() - start
        df.loc[chunksize, "Duration (s)"] = duration
        print(df)
    print(df)
    return df
