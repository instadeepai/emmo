"""Test the MoDec output parser."""
from __future__ import annotations

import emmo.io.external_tool_parsing as parser
from emmo.constants import REPO_DIRECTORY


folder = REPO_DIRECTORY / "validation" / "local" / "MoDec_example_output_set2"

# number of classes/motifs
N = 4

class_weights = parser.get_class_weight_from_responsibilities_modec(folder, N)
print(class_weights)

PWMs = parser.parse_pwms_modec(folder, N)
print(PWMs[0])

sequences = parser.parse_sequences_modec(folder, N)
print(sequences[:5], len(sequences))

LL = parser.loglikelihood_modec(sequences, PWMs, class_weights)
print("Log likelihood of MoDec model:", LL)
# best EMMo run (out of 20) is at the moment:
# Estimating frequencies (run 10),  470 EM steps,  logL = -47376.59041841925 ...
# finished 2000 sequences in 404.5923 seconds.
