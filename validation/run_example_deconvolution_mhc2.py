"""Run example deconvolution for MHC2."""
from __future__ import annotations

from emmo.constants import REPO_DIRECTORY
from emmo.em.mhc2 import EMRunnerMHC2
from emmo.pipeline.sequences import SequenceManager

input_name = "HLA-A0101_A0218_background_class_II"
directory = REPO_DIRECTORY / "validation" / "local"
file = directory / f"{input_name}.txt"
output_directory = directory / input_name

sm = SequenceManager.load_from_txt(file)
em_runner = EMRunnerMHC2(sm, 9, 2, "MHC2_biondeep")
em_runner.run(output_directory, output_all_runs=True, force=True)
