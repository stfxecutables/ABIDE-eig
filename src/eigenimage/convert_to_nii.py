import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
import sys
from ants import image_read
from pathlib import Path
from typing import List
import pandas as pd
import nibabel as nib
from tqdm.contrib.concurrent import process_map
from nibabel.nifti1 import Nifti1Image


sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants import NII_PATH
from src.eigenimage.compute import compute_eigenimage


SHAPES = NII_PATH / "shapes.json"

# we'll be cutting the timepoints to the last 176, which means computations
# take very close to 2 hours each time, so 11 computations per 24h window,
# i.e. 11 computations per batch
T_LENGTH = 176
BATCH_SIZE = 10


def get_files() -> List[Path]:
    df = pd.read_json(SHAPES)
    tr_2 = df.loc[df.dt == 2.0, :].sort_index()
    files = [NII_PATH / file for file in tr_2.index.to_list()]
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
