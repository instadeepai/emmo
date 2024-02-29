"""Simulate 9-mer sequences."""
from emmo.constants import NATURAL_AAS
from emmo.constants import REPO_DIRECTORY
from emmo.io.external_tool_parsing import parse_from_mhcmotifviewer
from emmo.pipeline.background import Background
from emmo.simulation.sequence_mixer import equal_length_frequencies


PWMs = REPO_DIRECTORY / "validation" / "PWMs_from_MHC_Motif_Viewer"
out_dir = REPO_DIRECTORY / "validation" / "local"
out_dir.mkdir(parents=True, exist_ok=True)

freq_matrices = [
    parse_from_mhcmotifviewer(PWMs / "HLA-A0101-PWM.txt", as_frequencies=True),
    parse_from_mhcmotifviewer(PWMs / "HLA-A0218-PWM.txt", as_frequencies=True),
    Background("uniprot").frequencies,
]

weights = [0.5, 0.3, 0.2]

freqs = freq_matrices[0]
for i in range(freqs.shape[0]):
    print(NATURAL_AAS[i], " ".join([str(round(x, 3)) for x in freqs[i]]))


with open(out_dir / "HLA-A0101_A0218_background.txt", "w") as f:
    for seq in equal_length_frequencies(1000, 9, freq_matrices, weights=weights):
        f.write(f"{seq}\n")
