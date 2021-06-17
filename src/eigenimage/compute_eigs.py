import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
from pathlib import Path

import numpy as np
from tqdm.contrib.concurrent import process_map

from src.eigenimage.compute import full_eigensignal
from src.eigenimage.compute_batch import get_files

DATA = Path(__file__).resolve().parent.parent.parent / "data"
MASK = DATA / "atlases/MASK.nii.gz"
EIGS_OUT = DATA / "eigs"
if not EIGS_OUT.exists():
    os.makedirs(EIGS_OUT)
N_PROCESSES = 84 if os.environ.get("CC_CLUSTER") == "niagara" else 8
OPTIMAL_CHUNKSIZE = 1


def compute_full_eigensignal(nii: Path) -> None:
    eigs = full_eigensignal(nii, MASK, covariance=True)
    extensions = "".join(nii.suffixes)
    outfile = EIGS_OUT / nii.name.replace(extensions, ".npy")
    np.save(outfile, eigs, allow_pickle=False)


if __name__ == "__main__":
    niis = get_files()
    process_map(compute_full_eigensignal, niis, max_workers=N_PROCESSES)
