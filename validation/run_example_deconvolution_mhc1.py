"""Run example deconvolution for MHC1."""
from __future__ import annotations

import numpy as np

from emmo.constants import REPO_DIRECTORY
from emmo.em.mhc1 import EMRunnerMHC1
from emmo.models.deconvolution import DeconvolutionModelMHC1
from emmo.pipeline.sequences import SequenceManager

input_name = "HLA-A0101_A0218_background_various_lengths"
directory = REPO_DIRECTORY / "validation" / "local"
file = directory / f"{input_name}.txt"
output_directory = directory / input_name

sm = SequenceManager.load_from_txt(file)
em_runner = EMRunnerMHC1(sm, 9, 2)
em_runner.run(output_directory, output_all_runs=True, force=True)

model = em_runner.best_model
for length, weights in model.class_weights.items():
    print(length, weights)

reloaded_model = DeconvolutionModelMHC1.load(output_directory)
for length, weights in reloaded_model.class_weights.items():
    print(length, weights)

print(np.allclose(model.ppm, reloaded_model.ppm))
