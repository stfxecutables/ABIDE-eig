# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent  # isort:skip
sys.path.append(str(ROOT))  # isort:skip
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

import os
import shutil
from pathlib import Path
from typing import List, Tuple, cast

import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.contrib.concurrent import process_map

from src.analysis.predict.reducers import subject_labels
from src.constants.environment import CC_CLUSTER
from src.constants.paths import DEEP, FULL_PREPROC_EIG, FULL_PREPROC_FMRI, PREPROC_EIG, PREPROC_FMRI

LABELS: List[int] = subject_labels(PREPROC_EIG)
FULL_LABELS: List[int] = subject_labels(FULL_PREPROC_FMRI)
SHAPE = (47, 59, 42, 175)

if CC_CLUSTER is None:  # reduce amount of subjects
    idx = next(
        StratifiedShuffleSplit(n_splits=1, train_size=150).split(
            FULL_PREPROC_FMRI, FULL_LABELS, FULL_LABELS
        )
    )[0]
    idx = np.array(idx, dtype=np.int32)
    FULL_PREPROC_FMRI = np.array(FULL_PREPROC_FMRI)[idx].tolist()
    FULL_LABELS = np.array(FULL_LABELS)[idx].tolist()


def copy(src_dest: Tuple[Path, Path]) -> None:
    src, dest = src_dest
    shutil.copyfile(src, dest)


def prepare_full_data(is_eigimg: bool = False) -> List[Path]:
    imgs = FULL_PREPROC_EIG if is_eigimg else FULL_PREPROC_FMRI
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir is None:  # nothing to do locally
        return cast(List[Path], imgs)

    # do not copy again
    data = Path(slurm_tmpdir).resolve() / "data"
    copies = [data / img.name for img in imgs]
    label = "EIG" if is_eigimg else "FMRI"
    copy_flag = data / f"{label}_COPIED.flag"
    if copy_flag.exists():
        return copies

    # actually copy when needed
    os.makedirs(data, exist_ok=True)
    print("Copying data files to $SLURM_TMPDIR...")
    process_map(copy, list(zip(imgs, copies)), disable=True)
    print("files copied.")
    copy_flag.touch()
    return copies


def prepare_data_files(is_eigimg: bool = False) -> List[Path]:
    imgs = PREPROC_EIG if is_eigimg else PREPROC_FMRI
    slurm_tmpdir = os.environ.get("SLURM_TMPDIR")
    if slurm_tmpdir is None:  # nothing to do locally
        return cast(List[Path], imgs)

    # do not copy again
    data = Path(slurm_tmpdir).resolve() / "data"
    copies = [data / img.name for img in imgs]
    label = "EIG" if is_eigimg else "FMRI"
    copy_flag = data / f"{label}_COPIED.flag"
    if copy_flag.exists():
        return copies

    # actually copy when needed
    os.makedirs(data, exist_ok=True)
    print("Copying data files to $SLURM_TMPDIR...")
    process_map(copy, list(zip(imgs, copies)), disable=True)
    print("files copied.")
    copy_flag.touch()
    return copies


def verify(fmri_eig: Tuple[Path, Path]) -> bool:
    fmri, eigimg = fmri_eig
    fail = False
    if not fmri.exists():
        print(f"Matching fMRI files currently missing: {fmri}")
        fail = True
    if not eigimg.exists():
        print(f"Matching fMRI files currently missing eigimg: {eigimg}")
        fail = True
    if not np.load(fmri).shape == SHAPE:
        print(f"Invalid input shape for file {fmri}")
        fail = True
    if not np.load(eigimg).shape == SHAPE:
        print(f"Invalid input shape for file {eigimg}")
        fail = True
    return fail


def verify_matching() -> None:
    print("Verifying fMRI files match eigimg files... ", end="", flush=True)
    n_unmatched = np.sum(
        process_map(
            verify,
            list(zip(PREPROC_FMRI, PREPROC_EIG)),
            total=len(PREPROC_EIG),
            desc="Verifying fMRI/eig matches",
        )
    )
    if n_unmatched > 0:
        print(f"Failure to verify. {n_unmatched} unmatched files.")


def get_testing_subsample() -> None:
    info = DataFrame(
        {"img": map(lambda p: p.name, PREPROC_EIG), "label": LABELS}, index=list(range(len(LABELS)))  # type: ignore # noqa
    )
    ctrl = info.loc[info["label"] == 0, :]
    auts = info.loc[info["label"] == 1, :]
    ctrl = ctrl.iloc[:50, :]
    auts = auts.iloc[:50, :]
    df = pd.concat([ctrl, auts], axis=0)
    df.to_csv(DEEP / "subjs.csv")
    sys.exit()
