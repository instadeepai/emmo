"""Run example deconvolution for MHC2 without offset weights."""
from __future__ import annotations

from emmo.constants import REPO_DIRECTORY
from emmo.em.experimental_modifications.mhc2_no_offset_weights import VariableLengthEM
from emmo.pipeline.sequences import SequenceManager


input_name = "HLA-A0101_A0218_background_class_II"
directory = REPO_DIRECTORY / "validation" / "local"
file = directory / f"{input_name}.txt"
output_directory = directory / f"{input_name}_no_class_weights"

sm = SequenceManager.load_from_txt(file)
em_runner = VariableLengthEM(sm, 9, 2, "MHC2_biondeep")
em_runner.run(output_directory, output_all_runs=True, force=True)
