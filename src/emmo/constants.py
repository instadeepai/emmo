"""Module used to define useful constants for the project."""
from __future__ import annotations

from emmo import REPO_DIRECTORY

# Constants related to GS and S3 buckets
DATA_BUCKET_NAME = "biondeep-data"
MODELS_BUCKET_NAME = "biondeep-models"
GS_BUCKET_PREFIX = "gs://"
S3_BUCKET_PREFIX = "s3://"
BUCKET_PREFIXES = (GS_BUCKET_PREFIX, S3_BUCKET_PREFIX)

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

DATA_DIRECTORY = REPO_DIRECTORY / "data"
MODELS_DIRECTORY = REPO_DIRECTORY / "models"
AVAILABLE_MODEL_DIRECTORIES = [
    MODELS_DIRECTORY / subdir for subdir in ("binding_predictor", "cleavage")
]
