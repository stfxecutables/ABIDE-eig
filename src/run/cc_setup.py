import logging
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
LOGDIR = ROOT / "logs"
RESULTS = ROOT / "results"


def setup(caller: str) -> Path:
    if os.environ.get("CC_CLUSTER") is not None:
        SCRATCH = os.environ["SCRATCH"]
        os.environ["MPLCONFIGDIR"] = str(Path(SCRATCH) / ".mplconfig")
    if not LOGDIR.exists():
        os.makedirs(LOGDIR, exist_ok=True)
    if not RESULTS.exists():
        os.makedirs(RESULTS, exist_ok=True)
    LOGFILE = LOGDIR / f"ERRORS_{caller}_{time.strftime('%b-%d__%H:%M:%S')}.log"
    logging.basicConfig(filename=LOGFILE, level=logging.DEBUG)
    return LOGFILE
