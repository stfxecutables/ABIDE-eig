import os
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.constants import NII_PATH  # noqa
from src.eigenimage.compute_batch import get_files, BATCH_SIZE

RUNTIME = "24:00:00"
JOBSCRIPT_OUTDIR = Path(__file__).resolve().parent.parent.parent / "job_scripts"
if not JOBSCRIPT_OUTDIR.exists():
    os.makedirs(JOBSCRIPT_OUTDIR, exist_ok=True)
CC_CLUSTER = os.environ.get("CC_CLUSTER")

# NOTE: --array is inclusive on both ends
HEADER_COMMON = """#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
#SBATCH --time={time}
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}%A__%a.out
#SBATCH --array=0-{array_end}
"""
if CC_CLUSTER == "niagara":
    RESOURCES = """#SBATCH --nodes=1
#SBATCH --ntasks-per-node=80
"""
elif CC_CLUSTER is None:  # local, testing
    RESOURCES = """#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
"""
else:
    raise EnvironmentError("Compute Canada cluster unsupported.")

HEADER = HEADER_COMMON + RESOURCES

SCRIPT = """
PROJECT=$SCRATCH/def-jlevman/dberger/ABIDE-eig
SCRIPT=$PROJECT/src/analysis/compute_batch.py

module load python/3.8.2
cd $SLURM_TMPDIR
tar -xf $PROJECT/venv.tar .venv
source .venv/bin/activate
PYTHON=$(which python)

echo "Job starting at $(date)"
$PYTHON $SCRIPT --batch=$SLURM_ARRAY_TASK_ID && \\
echo "Job done at $(date)"
"""


def uncomputed(nii: Path) -> bool:
    extensions = "".join(nii.suffixes)
    eigimg = nii.parent / f"{nii.name.replace(extensions, '_eigimg')}{extensions}"
    return not eigimg.exists()


def images_to_compute() -> int:
    return len(list(filter(uncomputed, NII_PATH.rglob("*minimal.nii.gz"))))


def compute_job_array_end_mixed() -> int:
    """
    Notes
    -----
    The way to do this correctly is get a good robust estimate of the compute time for each shape
    we have, then have a big DataFrame or dict of filename/estimated runtime pairs, and then just
    append runs until the expected time of a job is near 24h, and then that increases the number of
    arrays needed.
    """

    def hours(t: int) -> float:
        # multiply all time estimates by 1.1 and round up a bit
        if t == 176:
            return 2.00
        if t == 236:
            return 1.90
        if t == 316:
            return 2.35
        else:
            return 1.35

    shapes = pd.read_json(NII_PATH / "shapes.json").drop(columns=["H", "W", "D"])
    shapes.sort_values(by="T", ascending=False, inplace=True)  # do largest first
    hs = shapes["T"].apply(hours)
    batches, batch, lengths = [], [], []
    batch_length = i = 0
    while i < len(shapes):
        if batch_length + hs[i] < 24:
            batch.append(shapes.index[i])  # add file to batch
            batch_length += hs[i]
        else:  # batch is long enough, save it and reset
            batches.append(batch)
            lengths.append(batch_length)
            batch = []
            batch_length = 0
        i += 1
        if i == len(shapes):  # finish case
            batches.append(batch)
    array_end = len(batches) - 1  # SBATCH --array=0-3 runs 0-3 inclusive, i.e. 4 jobs
    return array_end


def compute_job_array_end() -> int:
    """
    Notes
    -----
    The way to do this correctly is get a good robust estimate of the compute time for each shape
    we have, then have a big DataFrame or dict of filename/estimated runtime pairs, and then just
    append runs until the expected time of a job is near 24h, and then that increases the number of
    arrays needed.
    """
    files = get_files()
    n_batches = len(files) // BATCH_SIZE  # 176 timepoints == 2.0 hours per image
    if len(files) % BATCH_SIZE > 0:  # remainders
        n_batches += 1
    return int(n_batches - 1)  # SBATCH --array=0-3 runs 0-3 inclusive, i.e. 4 jobs


def generate_script(script_outdir: Path = JOBSCRIPT_OUTDIR) -> str:
    job_name = "eigimg"
    header = HEADER.format(time=RUNTIME, job_name=job_name, array_end=compute_job_array_end())
    script = f"{header}{SCRIPT}"
    out = script_outdir / f"submit_{job_name}.sh"
    with open(out, mode="w") as file:
        file.write(script)
    print(f"Saved job script to {out}")
    return script


if __name__ == "__main__":
    # print(images_to_compute())
    generate_script()
