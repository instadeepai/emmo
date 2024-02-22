"""Module for motif related functions."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from emmo.constants import AA2IDX
from emmo.resources.background_freqs import get_background
from emmo.resources.substitution_matrix import blosum62_matrix

# ungapped lambda used by PSI-BLAST
UNGAPPED_LAMBDA = 0.3176


def count_amino_acids(sequences: str | Iterable[str]) -> list[int]:
    """Count the occurrences of all standard amino acids.

    Args:
        sequences: One or multiple sequences consisting of the 20 standard amino acids.

    Raises:
        ValueError: If an unknown character occurs in a sequence.

    Returns:
        A list of amino acid counts.
    """
    if isinstance(sequences, str):
        sequences = [sequences]

    counts = len(AA2IDX) * [0]

    for seq in sequences:
        for amino_acid in seq:
            try:
                counts[AA2IDX[amino_acid]] += 1
            except KeyError:
                raise ValueError(f"found unknown amino acid '{amino_acid}' in sequence: {seq}")

    return counts


def total_frequencies(sequences: str | Iterable[str]) -> np.array:
    """Compute the total frequencies of all standard amino acids.

    Args:
        sequences: One or multiple sequences consisting of the 20 standard amino acids.

    Raises:
        ValueError: If the sum of all amino acid counts is zero.

    Returns:
        An array of amino acid frequencies.
    """
    counts = np.array(count_amino_acids(sequences), dtype=np.float64)
    sum_ = counts.sum()

    if sum_ <= 0.0:
        raise ValueError("sum of all amino acid counts must be greater than zero")

    return counts / sum_


def pseudocount_frequencies(observed_frequencies: np.ndarray) -> np.ndarray:
    """Compute the pseudocount frequencies.

    Args:
        observed_frequencies: The observed frequencies.

    Returns:
        Pseudocount frequencies.

    References:
        [1] Altschul, S.F., Madden, T.L., Schäffer, A.A., Zhang, J., Zhang, Z., Miller, W. &
            Lipman, D.J. (1997) "Gapped BLAST and PSI-BLAST: a new generation of protein database
            search programs." Nucleic Acids Res. 25:3389-3402.
    """
    p = get_background("psi_blast")
    q = _target_frequencies()

    if p.shape[0] != observed_frequencies.shape[-1]:
        raise ValueError(
            "observed_frequencies and background do not match "
            f"(got {observed_frequencies.shape[-1]} and {p.shape[0]})"
        )

    if len(observed_frequencies.shape) == 1:
        # g_i = sum_j (f_i * q_ij) / p_j
        g = np.sum(observed_frequencies * q / p, axis=1)

    elif len(observed_frequencies.shape) == 2:
        g = np.zeros_like(observed_frequencies)
        for k in range(observed_frequencies.shape[0]):
            # g_i = sum_j (f_i * q_ij) / p_j
            g[k] = np.sum(observed_frequencies[k] * q / p, axis=1)

    else:
        raise ValueError("only 1- or 2-dimensional arrays are supported")

    return g


def frequencies_corrected_with_pseudocounts(
    observed_frequencies: np.ndarray, alpha: int | float, beta: int | float
) -> np.ndarray:
    """Estimate the amino acid frequencies using pseudocounts.

    Args:
        observed_frequencies: Observed amino acid frequecies.
        alpha: Weight for the observed frequencies.
        beta: Weight for the pseudocount frequencies.

    Returns:
        The corrected frequencies.

    References:
        [1] Altschul, S.F., Madden, T.L., Schäffer, A.A., Zhang, J., Zhang, Z., Miller, W. &
            Lipman, D.J. (1997) "Gapped BLAST and PSI-BLAST: a new generation of protein database
            search programs." Nucleic Acids Res. 25:3389-3402.
    """
    g = pseudocount_frequencies(observed_frequencies)

    return (alpha * observed_frequencies + beta * g) / (alpha + beta)


def position_probability_matrix(
    aligned_seqs: list[str], use_pseudocounts: bool = True, pseudocount_beta: int | float = 200
) -> np.ndarray:
    """Generate a position probability matrix from aligned (gapless) sequences.

    Args:
        aligned_seqs: The aligned and gapless sequences.
        use_pseudocounts: Whether to correct the probabilities with pseudocounts.
        pseudocount_beta: The weight of the pseudocounts.

    Returns:
        The position probability matrix.
    """
    motif_length = len(next(iter(aligned_seqs)))

    counts = np.zeros((motif_length, len(AA2IDX)), dtype=np.float64)

    for seq in aligned_seqs:
        if len(seq) != motif_length:
            raise ValueError("sequences must have the same length")

        for i, a in enumerate(seq):
            counts[i, AA2IDX[a]] += 1

    frequencies = counts / np.sum(counts, axis=1, keepdims=True)

    if use_pseudocounts:
        alpha = len(aligned_seqs)
        frequencies = frequencies_corrected_with_pseudocounts(frequencies, alpha, pseudocount_beta)

    # renormalize since the result does not sum to exactly 1
    frequencies = frequencies / np.sum(frequencies, axis=1, keepdims=True)

    return frequencies


def information_content(
    observed_frequencies: np.ndarray, background_frequencies: np.ndarray
) -> np.ndarray:
    """Position-wise information content of a position-probability matrix.

    Args:
        observed_frequencies: Observed frequencies.
        background_frequencies: Background frequencies.

    Returns:
        Information content.
    """
    if observed_frequencies.shape[-1] != background_frequencies.shape[-1]:
        raise ValueError("frequencies and background shapes do not match")

    return np.sum(
        observed_frequencies * np.log2(observed_frequencies / background_frequencies), axis=-1
    )


def _target_frequencies() -> np.ndarray:
    """Compute the substitution target frequencies for pseudocount estimation.

    These are necessary to compute the pseudocount frequencies.

    Returns:
        Substitution target frequencies for pseudocount estimation.

    References:
        [1] Altschul, S.F., Madden, T.L., Schäffer, A.A., Zhang, J., Zhang, Z., Miller, W. &
            Lipman, D.J. (1997) "Gapped BLAST and PSI-BLAST: a new generation of protein database
            search programs." Nucleic Acids Res. 25:3389-3402.
        [2] https://www.ncbi.nlm.nih.gov/BLAST/tutorial/Altschul-3.html
    """
    # background frequencies, this value must be the one corresponding to UNGAPPED_LAMBDA, so do
    # not change separately
    p = get_background("psi_blast")

    # q_ij = p_i * p_j * e ^ (UNGAPPED_LAMBDA * s_ij)
    return p[:, np.newaxis] * p[np.newaxis, :] * np.exp(UNGAPPED_LAMBDA * blosum62_matrix)


if __name__ == "__main__":
    TEST_SEQUENCES = [
        "MADSRDPASD",
        "QMQHWKEQRA",
        "AQKADVLTTG",
        "AGNPVGDKLN",
        "VITVGPRGPL",
        "LVQDVVFTDE",
        "MAHFDRERIP",
        "ERVVHAKGAG",
    ]

    print(position_probability_matrix(TEST_SEQUENCES, use_pseudocounts=False).tolist())
