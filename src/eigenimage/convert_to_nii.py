import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import sys
from pathlib import Path
from typing import List

import nibabel as nib
import pandas as pd
from ants import image_read
from nibabel.nifti1 import Nifti1Image
from tqdm.contrib.concurrent import process_map

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants.paths import NIIS
from src.eigenimage.compute import compute_eigenimage

SHAPES = NIIS / "shapes.json"

# we'll be cutting the timepoints to the last 176, which means computations
# take very close to 2 hours each time, so 11 computations per 24h window,
# i.e. 11 computations per batch
T_LENGTH = 176
BATCH_SIZE = 10


def get_files() -> List[Path]:
    df = pd.read_json(SHAPES)
    tr_2 = df.loc[df.dt == 2.0, :].sort_index()
    files = [NIIS / file for file in tr_2.index.to_list()]
    return files


def save_eig_nii(file: Path) -> None:
    extensions = "".join(file.suffixes)
    out = Path(f"{str(file).replace(extensions, '')}_eigimg.nii.gz")
    nii: Nifti1Image = image_read(str(file)).to_nibabel()
    eigimg = compute_eigenimage(file, covariance=True, t=T_LENGTH)
    new = Nifti1Image(
        dataobj=eigimg,
        affine=nii.affine,
        header=nii.header,
        extra=nii.extra,
    )
    nib.save(new, str(out))
    print(f"Saved eigenimage to {out}.")


if __name__ == "__main__":
    files = get_files()
    process_map(save_eig_nii, files)
