"""Package unsupervised peptide-MHC binding/presentation deconvolution and prediction.

The package contains Python implementations of the methods MixMHCp for MHC1 deconvolution (Gfeller
et al. 2018, J Immunol), MoDec for MHC2 deconvolution (Racle et al. 2019, Nat Biotechnol), and
MixMHC2pred for MHC2 binding prediction (Racle et al. 2019, Nat Biotechnol), with some
modifications, simplifications, and improvements.
"""
import os
from pathlib import Path

from dotenv import load_dotenv

# Package information
__version__ = "0.0.1"

CURRENT_DIRECTORY = Path(__file__).resolve().parent
REPO_DIRECTORY = CURRENT_DIRECTORY.parent.parent

os.environ["REPO_DIRECTORY"] = str(REPO_DIRECTORY)

# Set the variables defined in .env as environment variables
load_dotenv(dotenv_path=REPO_DIRECTORY / ".env")
