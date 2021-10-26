import os
from pathlib import Path

CC_CLUSTER = os.environ.get("CC_CLUSTER")
SLURM_TMPDIR = os.environ.get("SLURM_TMPDIR")
SCRATCH = os.environ["SCRATCH"]
if CC_CLUSTER is not None and (CC_CLUSTER == "niagara"):
    SCRATCH = os.environ["SCRATCH"]
    os.environ["MPLCONFIGDIR"] = str(Path(SCRATCH) / ".mplconfig")
