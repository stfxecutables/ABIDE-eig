import os
from pathlib import Path

os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
if os.environ.get("CC_CLUSTER") == "niagara":
    os.environ["MPLCONFIGDIR"] = str(Path(os.environ["SCRATCH"]) / ".mplconfig")

import sys
from argparse import ArgumentParser

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
>>> np.unique(df["T"], return_counts=True)
(array([116, 146, 176, 196, 202, 236, 246, 296, 316]), array([ 60,  46, 208,  60,   1,  13,  27, 128,  13]))
>>> np.unique(df["dt"].round(2), return_counts=True)
(array([1.5 , 1.66, 1.67, 2.  , 2.2 , 3.  ]), array([ 69,   1,  26, 396,   4,  60]))
>>> np.unique(df["dt"].round(1), return_counts=True)
(array([1.5, 1.7, 2. , 2.2, 3. ]), array([ 69,  27, 396,   4,  60]))

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

Estimating time for subject with T = 116 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 146 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 176 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 196 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 202 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 236 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 246 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 296 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 316 using 4245 voxels. Using 84 processes.

                                             Estimated Time
UCLA_1_0051201_func_minimal.nii.gz   0 days 01:05:31.390913
Trinity_0050232_func_minimal.nii.gz  0 days 01:44:45.060964
NYU_0050952_func_minimal.nii.gz      0 days 02:00:23.966623
Pitt_0050003_func_minimal.nii.gz     0 days 01:58:41.036831
CMU_a_0050647_func_minimal.nii.gz    0 days 02:25:18.577688
CMU_a_0050642_func_minimal.nii.gz    0 days 02:48:55.975521
Leuven_1_0050682_func_minimal.nii.gz 0 days 03:26:43.577640
UM_1_0050272_func_minimal.nii.gz     0 days 04:39:24.942342
CMU_b_0050643_func_minimal.nii.gz    0 days 03:42:51.118297

We have 396 subjects so far where the TR is 2.0. For those subjects, the number of timepoints and
counts is:

  T   N_subj
146       46
176      208
202        1
236       13
296      128

If we cut the subjects with more than 176 timepoint down to their last 176 timepoints we get more
data, it becomes more comparable, and is faster to compute (about 2 hours per scan).

IF WE USE MASKS:

Estimating time for subject with T = 116 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 146 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 176 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 196 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 202 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 236 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 246 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 296 using 4245 voxels. Using 84 processes.
Estimating time for subject with T = 316 using 4245 voxels. Using 84 processes.
                                             Estimated Time
UCLA_1_0051201_func_minimal.nii.gz   0 days 00:08:32.544845
Trinity_0050232_func_minimal.nii.gz  0 days 00:11:10.752227
NYU_0050952_func_minimal.nii.gz      0 days 00:14:18.542031  ***
Pitt_0050003_func_minimal.nii.gz     0 days 00:16:11.987535
CMU_a_0050647_func_minimal.nii.gz    0 days 00:16:59.682331
CMU_a_0050642_func_minimal.nii.gz    0 days 00:21:22.091502
Leuven_1_0050682_func_minimal.nii.gz 0 days 00:22:19.419580
UM_1_0050272_func_minimal.nii.gz     0 days 00:33:08.536620
CMU_b_0050643_func_minimal.nii.gz    0 days 00:35:49.699683



"""

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--decimation", type=int, default=128)
    decimation = parser.parse_args().decimation
    df = DataFrame()
    for nii in TEST_NIIS:
        duration = estimate_computation_time(
            nii, covariance=True, estimate_time=True, decimation=decimation
        )
        df.loc[nii.name, "Estimated Time"] = duration
    print(df)
