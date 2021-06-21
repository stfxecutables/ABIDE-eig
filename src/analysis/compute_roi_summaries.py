import os
import sys
from pathlib import Path

if os.environ.get("CC_CLUSTER") is not None:
    SCRATCH = os.environ["SCRATCH"]
    os.environ["MPLCONFIGDIR"] = str(Path(SCRATCH) / ".mplconfig")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.analysis.rois import (
    precompute_all_eigimg_roi_reductions,
    precompute_all_func_roi_reductions,
)

if __name__ == "__main__":
    # precompute_all_func_roi_reductions()
    precompute_all_eigimg_roi_reductions()
    # sys.exit()
