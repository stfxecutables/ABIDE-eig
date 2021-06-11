import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
NII_PATH = DATA / "niis"
CKPT_PATH = DATA / "ckpts"
if not CKPT_PATH.exists():
    os.makedirs(CKPT_PATH, exist_ok=True)
