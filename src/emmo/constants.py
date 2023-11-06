"""Module used to define useful constants for the project."""
from __future__ import annotations

from pathlib import Path


NATURAL_AAS: tuple[str, ...] = (
    "A",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "K",
    "L",
    "M",
    "N",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "V",
    "W",
    "Y",
)
AA2IDX = {aa: idx for idx, aa in enumerate(NATURAL_AAS)}
NUM_AAS = len(NATURAL_AAS)

CURRENT_DIRECTORY = Path(__file__).resolve().parent
REPO_DIRECTORY = CURRENT_DIRECTORY.parent.parent
MODEL_DIRECTORY = REPO_DIRECTORY / "models"

if __name__ == "__main__":
    print("Package path:", CURRENT_DIRECTORY)
    print("Repository path:", REPO_DIRECTORY)
