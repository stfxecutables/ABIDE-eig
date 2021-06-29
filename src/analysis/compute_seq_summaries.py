import os
import sys
from pathlib import Path

if os.environ.get("CC_CLUSTER") is not None:
    SCRATCH = os.environ["SCRATCH"]
    os.environ["MPLCONFIGDIR"] = str(Path(SCRATCH) / ".mplconfig")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.analysis.predict.reducers import eigvals, max, mean, median, pca, std
from src.analysis.predict.sequence import compute_sequence_reductions

if __name__ == "__main__":
    # for reducer in [eigvals, pca, mean, std, median, max]:
    for reducer in [eigvals]:
        for norm in ["div", "diff", None]:
            print(f"Computing sequence reductions using norm={norm} and reducer {reducer.__name__}")
            compute_sequence_reductions(
                source="func",
                norm=norm,
                reducer=reducer,
            )
    # sys.exit()
