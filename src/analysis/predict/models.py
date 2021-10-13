# fmt: off
import sys  # isort:skip
from pathlib import Path  # isort:skip
ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(ROOT))
# fmt: on

import optuna

from src.analysis.features import FEATURES, NormMethod
from src.analysis.predict.hypertune import evaluate_hypertuned, hypertune_classifier

"""
Notes
-----
We have a few feature types:

* multi-channel 1D
    - ROI means, ROI sds, their concatenation
* 2D matrix
    - r_mean, r_sd, and their concatenation
* 1D non-sequential (single or multi-channel)
    - r_mean, r_sd, r_desc (multichannel)
    - Lap, Lap_concat, eig_r_mean, eig_r_sd, and concat
* 1D sequential
    - Laplacian and plain eigenvalues, and their concatenations

Classic ML: SVM, LR, RF AdaBoostDTree
Deep Learn: MLP, Conv1D, ...?
"""

if __name__ == "__main__":
    norm = NormMethod.S_MINMAX
    features = [f for f in FEATURES if len(f.shape_data.shape) == 1]
    for f in features:
        print(f"Fitting {f} using normalization method {norm}")
        x, y = f.load(norm)
        htune_result = hypertune_classifier("rf", x, y, n_trials=10, verbosity=optuna.logging.DEBUG)
        result = evaluate_hypertuned(htune_result, 5, x, y, log=True)
        print("Validation results:")
        for key, val in result.items():
            print(f"{key}: {val}")
