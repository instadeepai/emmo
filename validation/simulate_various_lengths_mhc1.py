"""Simulate MHC1 ligands of variable length."""
from emmo.constants import NATURAL_AAS
from emmo.constants import REPO_DIRECTORY
from emmo.io.external_tool_parsing import parse_from_mhcmotifviewer
from emmo.resources.background_freqs import get_background
from emmo.simulation.sequence_mixer import equal_length_frequencies
from emmo.simulation.sequence_mixer import modify_lengths_fixed_termini


PWMs = REPO_DIRECTORY / "validation" / "PWMs_from_MHC_Motif_Viewer"
out_dir = REPO_DIRECTORY / "validation" / "local"
out_dir.mkdir(parents=True, exist_ok=True)

freq_matrices = [
    parse_from_mhcmotifviewer(PWMs / "HLA-A0101-PWM.txt", as_frequencies=True),
    parse_from_mhcmotifviewer(PWMs / "HLA-A0218-PWM.txt", as_frequencies=True),
    get_background(which="uniprot"),
]

weights = [0.5, 0.3, 0.2]

freqs = freq_matrices[0]
for i in range(freqs.shape[0]):
    print(NATURAL_AAS[i], " ".join([str(round(x, 3)) for x in freqs[i]]))

seqs = equal_length_frequencies(2000, 9, freq_matrices, weights=weights)

lengths = [8, 9, 10, 11, 12]
lweights = [2, 5, 3, 2, 1]
seqs = modify_lengths_fixed_termini(seqs, lengths, lweights, get_background(which="uniprot"))


with open(out_dir / "HLA-A0101_A0218_background_various_lengths.txt", "w") as f:
    for seq in seqs:
        f.write(f"{seq}\n")
