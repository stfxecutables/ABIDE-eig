from dataclasses import dataclass
from pathlib import Path


@dataclass
class Atlas:
    name: str
    path: Path
    legend: Path
