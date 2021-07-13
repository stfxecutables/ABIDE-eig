# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(ROOT))
from src.run.cc_setup import setup_environment  # isort:skip
setup_environment()
# fmt: on


import sys
import traceback
from pathlib import Path

from sklearn.model_selection import ParameterGrid

from src.analysis.predict.reducers import eigvals, max, mean, pca, std
from src.analysis.predict.sequence import compute_sequence_reductions

# NOTE: given the GRID
#
#    GRID = dict(
#        source=["func", "eigimg"],
#        norm=["div", "diff", None],
#        reducer=[pca, mean, max, std, eigvals],
#    )
#
# expected runtime is well under an hour on Niagara.
if __name__ == "__main__":
    GRID = dict(
        source=["func", "eigimg"],
        norm=["div", "diff", None],
        reducer=[pca, mean, max, std, eigvals],
    )
    for args in list(ParameterGrid(GRID)):
        try:
            compute_sequence_reductions(**args)
        except Exception as e:
            print(f"Got exception {e}")
            traceback.print_exc()
