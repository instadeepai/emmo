"""Run example deconvolution for MHC1."""
from __future__ import annotations

from emmo.constants import REPO_DIRECTORY
from emmo.em.experimental_modifications.mhc1_mixmhcp import FullEM
from emmo.pipeline.sequences import SequenceManager


# input_name = 'HLA-A0101_A0218_background'
input_name = "HLA-A0101_A0218_background_various_lengths"
directory = REPO_DIRECTORY / "validation" / "local"
file = directory / f"{input_name}.txt"
output_directory = directory / input_name

sm = SequenceManager.load_from_txt(file)
em_runner = FullEM(sm, 9, 2)
em_runner.run()

em_runner.write_results(output_directory, force=True)
