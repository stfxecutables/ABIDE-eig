import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"

import sys
from pathlib import Path

from pandas import DataFrame

sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.constants import NII_PATH
from src.eigenimage.compute import estimate_computation_time

TEST_NIIS = [
    NII_PATH / "NYU_0051015_func_minimal.nii.gz",  # T = 176
    NII_PATH / "CMU_a_0050642_func_minimal.nii.gz",  # T = 236
    NII_PATH / "CMU_b_0050643_func_minimal.nii.gz",  # T = 316
]

if __name__ == "__main__":
    df = DataFrame()
    for nii in TEST_NIIS:
        duration = estimate_computation_time(
            nii, covariance=True, estimate_time=True, decimation=1024
        )
        df.loc[nii.name, "Estimated Time"] = duration
    print(df)