"""Module used to define useful constants for the project."""
from __future__ import annotations

from emmo import REPO_DIRECTORY


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

MODEL_DIRECTORY = REPO_DIRECTORY / "models"
