import os
import traceback
from pathlib import Path
from typing import Optional, Tuple

import nibabel as nib
from tqdm.contrib.concurrent import process_map

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
MASK = DATA / "atlases/MASK.nii.gz"
NIIS = DATA / "nii_cpac"
OUT = DATA / "nii_cpac_f16"
if not OUT.exists():
    os.makedirs(OUT, exist_ok=True)


def to_masked_float16(nii: Path) -> Optional[Tuple[Path, str]]:
    try:
        img = nib.load(str(nii))
        air = ~nib.load(str(MASK)).get_fdata().astype(bool)
        data = img.get_fdata().astype("float16")
        data[air] = 0
        new = nib.Nifti2Image(data, affine=img.affine, header=img.header, extra=img.extra)
        outfile = OUT / nii.name
        nib.save(new, str(outfile))
        return None
    except Exception as e:
        traceback.print_exc()
        print(nii, e)
        return nii, str(e)


if __name__ == "__main__":
    niis = sorted(NIIS.rglob("*.nii.gz"))
    results = process_map(to_masked_float16, niis, desc="Converting to float16")
    fails = [path for path in results if path is not None]
    if len(fails) > 0:
        print("Failed to convert files:")
        for file, message in fails:
            print(f"{file.name} - ERROR {message}")
