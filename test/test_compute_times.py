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
    NII_PATH / "UCLA_1_0051201_func_minimal.nii.gz",  # 116
    NII_PATH / "Trinity_0050232_func_minimal.nii.gz",  # 146
    NII_PATH / "NYU_0050952_func_minimal.nii.gz",  # 176
    NII_PATH / "Pitt_0050003_func_minimal.nii.gz",  # 196
    NII_PATH / "CMU_a_0050647_func_minimal.nii.gz",  # 202
    NII_PATH / "CMU_a_0050642_func_minimal.nii.gz",  # 236
    NII_PATH / "Leuven_1_0050682_func_minimal.nii.gz",  # 246
    NII_PATH / "UM_1_0050272_func_minimal.nii.gz",  # 296
    NII_PATH / "CMU_b_0050643_func_minimal.nii.gz",  # 316
]

"""
>>> df = pd.read_json("shapes.json")
>>> df.iloc[np.unique(df["T"], return_index=True)[1], :]
                                       H   W   D    T       dt
UCLA_1_0051201_func_minimal.nii.gz    61  73  61  116  3.00000
Trinity_0050232_func_minimal.nii.gz   61  73  61  146  2.00000
NYU_0050952_func_minimal.nii.gz       61  73  61  176  2.00000
Pitt_0050003_func_minimal.nii.gz      61  73  61  196  1.50000
CMU_a_0050647_func_minimal.nii.gz     61  73  61  202  2.00000
CMU_a_0050642_func_minimal.nii.gz     61  73  61  236  2.00000
Leuven_1_0050682_func_minimal.nii.gz  61  73  61  246  1.66665
UM_1_0050272_func_minimal.nii.gz      61  73  61  296  2.00000
CMU_b_0050643_func_minimal.nii.gz     61  73  61  316  1.50000
"""

if __name__ == "__main__":
    df = DataFrame()
    for nii in TEST_NIIS:
        duration = estimate_computation_time(
            nii, covariance=True, estimate_time=True, decimation=1024
        )
        df.loc[nii.name, "Estimated Time"] = duration
    print(df)
