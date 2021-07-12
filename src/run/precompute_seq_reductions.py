import os
import sys
import traceback
from pathlib import Path

from sklearn.model_selection import ParameterGrid

if os.environ.get("CC_CLUSTER") is not None:
    SCRATCH = os.environ["SCRATCH"]
    os.environ["MPLCONFIGDIR"] = str(Path(SCRATCH) / ".mplconfig")
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.analysis.predict.reducers import eigvals, max, mean, pca, std
from src.analysis.predict.sequence import compute_sequence_reductions

if __name__ == "__main__":
    GRID = dict(
        source=["func", "eigimg"],
        norm=["div", "diff", None],
        reducer=[pca, mean, max, std, eigvals],
    )
    # expected runtime is well under an hour on Niagara
    for args in list(ParameterGrid(GRID)):
        try:
            compute_sequence_reductions(**args)
        except Exception as e:
            print(f"Got exception {e}")
            traceback.print_exc()
