import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
from argparse import ArgumentParser
from pathlib import Path

import nibabel as nib
from tqdm.contrib.concurrent import process_map

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants import NII_PATH
from src.eigenimage.compute import (
    compute_eigencontribution,
    eigimage_parallel,
    find_optimal_chunksize,
)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--optimize", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=32)
    args = parser.parse_args()
    nii = sorted(NII_PATH.rglob("*.nii.gz"))[0]
    img = nib.load(str(nii)).get_fdata()
    if args.optimize:
        df = find_optimal_chunksize(img, covariance=True)
        outfile = Path(__file__).resolve().parent.parent.parent / "chunsize_times.json"
        df.to_json(outfile, indent=2)
        print(f"Saved chunksize report to {outfile}")
    else:
        # 24h with 8 cores
        eigimage_parallel(img, chunksize=args.chunk_size, covariance=True)
        pass
