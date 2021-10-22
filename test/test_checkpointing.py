import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
from pathlib import Path

if os.environ.get("CC_CLUSTER") is not None:
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ["SCRATCH"]) / ".mplconfig")

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.constants.paths import NIIS
from src.eigenimage.compute import compute_eigenimage

TEST_NIIS = [
    NIIS / "NYU_0051015_func_minimal.nii.gz",  # T = 176
    NIIS / "CMU_a_0050642_func_minimal.nii.gz",  # T = 236
    NIIS / "CMU_b_0050643_func_minimal.nii.gz",  # T = 316
]

if __name__ == "__main__":
    compute_eigenimage(TEST_NIIS[0], covariance=True)
