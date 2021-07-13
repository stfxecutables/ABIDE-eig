# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.run.cc_setup import setup_environment  # isort:skip
setup_environment()
# fmt: on


import sys
from pathlib import Path

from src.analysis.rois import (
    precompute_all_eigimg_roi_reductions,
    precompute_all_func_roi_reductions,
)

if __name__ == "__main__":
    precompute_all_func_roi_reductions()
    precompute_all_eigimg_roi_reductions()
