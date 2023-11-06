"""Simulate MHC2 ligands of variable length."""
from emmo.constants import REPO_DIRECTORY
from emmo.io.external_tool_parsing import parse_from_mhcmotifviewer
from emmo.resources.background_freqs import get_background
from emmo.simulation.sequence_mixer import equal_length_frequencies
from emmo.simulation.sequence_mixer import modify_lengths_by_adding_flanks


PWMs = REPO_DIRECTORY / "validation" / "PWMs_from_MHC_Motif_Viewer"
out_dir = REPO_DIRECTORY / "validation" / "local"
out_dir.mkdir(parents=True, exist_ok=True)

freq_matrices = [
    parse_from_mhcmotifviewer(PWMs / "HLA-A0101-PWM.txt", as_frequencies=True),
    parse_from_mhcmotifviewer(PWMs / "HLA-A0218-PWM.txt", as_frequencies=True),
    get_background(which="uniprot"),
]

weights = [0.5, 0.3, 0.2]

seqs = equal_length_frequencies(2000, 9, freq_matrices, weights=weights)

lengths = list(range(9, 25))
n_lengths = len(lengths)
lweights = [1 / n_lengths for _ in range(n_lengths)]
seqs = modify_lengths_by_adding_flanks(seqs, lengths, lweights, get_background(which="uniprot"))

with open(out_dir / "HLA-A0101_A0218_background_class_II.txt", "w") as f:
    for seq in seqs:
        f.write(f"{seq}\n")
