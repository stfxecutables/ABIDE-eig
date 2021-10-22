# fmt: off
from pathlib import Path  # isort:skip
import sys  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent.parent
sys.path.append(str(ROOT))
# from src.run.cc_setup import setup_environment  # isort:skip
# setup_environment()
# fmt: on

import sys
from pathlib import Path
from typing import List

from src.constants.environment import CC_CLUSTER
from src.constants.paths import SCRIPT_OUTDIR

EIGIMG = "$PROJECT/data/deep/eigimg"
FMRI = "$PROJECT/data/deep/fmri"

RUNTIME = "24:00:00"

HEADER_COMMON = """#!/bin/bash
#SBATCH --account=def-jlevman
#SBATCH --time={time}
#SBATCH --signal=INT@300
#SBATCH --job-name={job_name}
#SBATCH --output={job_name}__%j.out
#SBATCH --mail-user=dberger@stfx.ca
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --mail-type=FAIL
"""
if CC_CLUSTER == "siku":
    RESOURCES = """#SBATCH --gres=gpu:v100:1
#SBATCH --partition=all_gpus
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
"""
    NUM_WORKERS = 8
elif CC_CLUSTER == "beluga":
    RESOURCES = """#SBATCH --gres=gpu:v100:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
"""
    NUM_WORKERS = 16
elif CC_CLUSTER == "cedar":
    RESOURCES = """#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
"""
    NUM_WORKERS = 8
elif CC_CLUSTER == "graham":
    RESOURCES = """#SBATCH --gres=gpu:t4:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --constraint=cascade
"""
    NUM_WORKERS = 16
# elif CC_CLUSTER == "graham":
#     RESOURCES = """#SBATCH --gres=gpu:v100:1
# #SBATCH --cpus-per-task=5
# #SBATCH --mem=4G
# """
else:
    raise EnvironmentError("Compute Canada cluster is unrecognized.")

HEADER = HEADER_COMMON + RESOURCES

SCRIPT = """
PROJECT=$HOME/projects/def-jlevman/dberger/ABIDE-eig
LOGS=$PROJECT/lightning_logs
FMRI=$PROJECT/data/deep/fmri
EIGIMG=$PROJECT/data/deep/eigimg

echo "Setting up python venv"
cd $SLURM_TMPDIR
tar -xf $PROJECT/venv.tar .venv
source .venv/bin/activate
PYTHON=$(which python)

echo "Job starting at $(date)"
tensorboard --logdir=$LOGS --host 0.0.0.0 &
{command} && \\
echo "Job done at $(date)"
"""


def generate_script(script_outdir: Path = SCRIPT_OUTDIR, is_eigimg: bool = False) -> str:
    lines: List[str]
    mlp_lines: List[str]

    pythonfile = "$PROJECT/src/analysis/predict/deep_learning/models/conv_lstm.py"
    args = " ".join(sys.argv[1:])
    job_name = "deep_eigimg" if is_eigimg else "deep_fmri"
    command = f"$PYTHON {pythonfile} {args}"
    header = HEADER.format(time=RUNTIME, job_name=job_name)

    script = f"{header}{SCRIPT.format(command=command)}"
    out = script_outdir / f"submit_{job_name}.sh"
    with open(out, mode="w") as file:
        file.write(script)
    print(f"Saved job script to {out}")

    return script


def generate_htune_scripts(script_outdir: Path = SCRIPT_OUTDIR, is_eigimg: bool = False) -> str:
    pythonfile = "$PROJECT/src/analysis/predict/deep_learning/optimize.py"
    args = " ".join(sys.argv[1:])
    job_name = "htune_eigimg" if is_eigimg else "htune_fmri"
    command = f"$PYTHON {pythonfile} {args} --num_workers={NUM_WORKERS}"
    header = HEADER.format(time=RUNTIME, job_name=job_name)

    script = f"{header}{SCRIPT.format(command=command)}"
    out = script_outdir / f"submit_{job_name}.sh"
    with open(out, mode="w") as file:
        file.write(script)
    print(f"Saved job script to {out}")

    return script


if __name__ == "__main__":
    print(f"Will save scripts in {SCRIPT_OUTDIR}")
    generate_script(is_eigimg=False)
    generate_script(is_eigimg=True)
    generate_htune_scripts(is_eigimg=False)
    generate_htune_scripts(is_eigimg=True)
