import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
from pathlib import Path

os.environ["MPLCONFIGDIR"] = str(Path(os.environ["SCRATCH"]) / ".mplconfig")

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.constants import NII_PATH
from src.eigenimage.compute import compute_eigenimage

TEST_NIIS = [
    NII_PATH / "NYU_0051015_func_minimal.nii.gz",  # T = 176
    NII_PATH / "CMU_a_0050642_func_minimal.nii.gz",  # T = 236
    NII_PATH / "CMU_b_0050643_func_minimal.nii.gz",  # T = 316
]

if __name__ == "__main__":
    compute_eigenimage(TEST_NIIS[0], covariance=True)
