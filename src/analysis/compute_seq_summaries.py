import os
import sys
from pathlib import Path

if os.environ.get("CC_CLUSTER") is not None:
    SCRATCH = os.environ["SCRATCH"]
    os.environ["MPLCONFIGDIR"] = str(Path(SCRATCH) / ".mplconfig")
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.analysis.predict.sequence import compute_sequence_reductions

if __name__ == "__main__":
    for norm in ["div", "diff", None]:
        print(f"Computing sequence reductions using norm={norm}")
        compute_sequence_reductions(
            source="func",
            norm=norm,
        )
    # sys.exit()
