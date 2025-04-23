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

MHC1_LENGTH_RANGE = (8, 14)

MHC2_BINDING_CORE_SIZE = 9

DATA_DIRECTORY = REPO_DIRECTORY / "data"
MODELS_DIRECTORY = REPO_DIRECTORY / "models"
AVAILABLE_MODEL_DIRECTORIES = [
    MODELS_DIRECTORY / subdir for subdir in ("binding_predictor", "cleavage")
]

# default column names
MHC1_ALLELE_COL = "allele"
MHC2_ALPHA_COL = "allele_alpha"
MHC2_BETA_COL = "allele_beta"

# lengths of the N- and C-terminal parts of MHC1 ligands that are considered most important
# for anchoring the peptide to the MHC1 molecule
MHC1_N_TERMINAL_ANCHORING_LENGTH = 3
MHC1_C_TERMINAL_ANCHORING_LENGTH = 2

# MHC1 N-terminal and C-terminal overhang penalties, default values as in MixMHCp
MHC1_N_TERMINAL_OVERHANG_PENALTY = 0.05
MHC1_C_TERMINAL_OVERHANG_PENALTY = 0.2
