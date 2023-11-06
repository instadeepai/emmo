"""Module for simulating MHC1 and MHC2 ligands."""
from __future__ import annotations

import numpy as np

from emmo.constants import NATURAL_AAS


def equal_length_frequencies(
    n: int, k: int, freq_matrices: list[np.ndarray], weights: list | np.ndarray | None = None
) -> list[str]:
    """Simulate n sequences of a given length k.

    The optional weights parameters determines how many sequences are generated for each motif. By
    default, an (approximate) equal number of motifs is generated for each motif.

    Args:
        n: Number of sequences to simulate.
        k: Sequence length.
        freq_matrices: Position probability matrices. Can also contain a flat motif.
        weights: Weights for the motifs. If this is not provided, a uniform distribution is used.

    Returns:
        The simulated sequences.
    """
    if not weights:
        weights = len(freq_matrices) * [1 / len(freq_matrices)]

    weights = np.asarray(weights)
    weights /= sum(weights)

    r = np.random.rand(n, k)
    sequences: list[str] = []

    for freq_index, freqs in enumerate(freq_matrices):
        if freq_index < len(freq_matrices) - 1:
            to_simulate = round(n * weights[freq_index])
        else:
            # ensure that we end up with exactly N sequences in the end, last motif may therefore
            # have one additional sequence
            to_simulate = n - len(sequences)

        # flat motif
        if len(freqs.shape) == 1:
            cum_freqs = np.cumsum(freqs)

            for i in range(len(sequences), len(sequences) + to_simulate):
                positions = [np.argmax(cum_freqs > r[i, j]) for j in range(k)]
                sequences.append("".join([NATURAL_AAS[p] for p in positions]))

        # normal motif
        else:
            cum_freqs = np.cumsum(freqs, axis=0)

            for i in range(len(sequences), len(sequences) + to_simulate):
                positions = [np.argmax(cum_freqs[:, j] > r[i, j]) for j in range(k)]
                sequences.append("".join([NATURAL_AAS[p] for p in positions]))

    return sequences


def _draw_weighted(n: int, weights: list | np.ndarray) -> np.ndarray:
    """Draw n times from [0, len(weights)] in a weighted manner.

    Args:
        n: Number of integers to draw.
        weights: The weights. This also determines the range of the integers to draw from.

    Returns:
        The drawn integers.
    """
    weights = np.asarray(weights, dtype=np.float64)
    weights /= sum(weights)
    cum_weights = np.cumsum(weights)

    return np.argmax(np.random.rand(n)[:, np.newaxis] < cum_weights, axis=1)


def _generate_piece(length: int, cum_freqs: np.ndarray) -> str:
    """Generate a random sequence piece from a common distribution.

    Args:
        length: Length of the sequence piece to be simulated.
        cum_freqs: Cumulative amino acid frequencies.

    Returns:
        The simulated sequence.
    """
    piece = ""

    for r in np.random.rand(length):
        a = np.argmax(cum_freqs > r)
        piece += NATURAL_AAS[a]

    return piece


def modify_lengths_fixed_termini(
    sequences: list[str],
    lengths: list[int],
    weights: list | np.ndarray,
    background_freqs: np.ndarray,
    fix_n_terminus: int = 3,
    fix_c_terminus: int = 2,
) -> list[str]:
    """Add or delete random sequence pieces within the given sequences.

    This function is intended for the simulation of MHC1 ligands of variable length. The N- and
    C-terminus of the input sequences are fixed and the newly simulated pieces are inserted
    somewhere in between, or pieces of the given sequence are removed.

    Args:
        sequences: Input sequences.
        lengths: The target lengths.
        weights: The weights/frequencies corresponding to the target lengths.
        background_freqs: The amino acid frequencies based on which the new sequence pieces are
            generated.
        fix_n_terminus: The length of the N-terminal sequence part to fix.
        fix_c_terminus: The length of the C-terminal sequence part to fix.

    Raises:
        ValueError: If the smallest target length is shorter that the terminal parts to fix.
        ValueError: If not all input sequences have the same length.

    Returns:
        The modified sequences.
    """
    # the length of the input sequences
    k = len(sequences[0])

    if min(lengths) < fix_n_terminus + fix_c_terminus:
        raise ValueError(f"smallest length {min(lengths)} is invalid")

    cum_background_freqs = np.cumsum(background_freqs)
    r = _draw_weighted(len(sequences), weights)

    modified_sequences = []
    for i, seq in enumerate(sequences):
        if len(seq) != k:
            raise ValueError("sequences do not have the same length")

        length = lengths[r[i]]

        if length == k:
            modified_sequences.append(seq)
        elif length < k:
            pos = np.random.randint(low=fix_n_terminus, high=length - fix_c_terminus + 1)
            modified_sequences.append(seq[:pos] + seq[pos + k - length :])
        else:
            pos = np.random.randint(low=fix_n_terminus, high=k - fix_c_terminus)
            piece = _generate_piece(length - k, cum_background_freqs)
            modified_sequences.append(seq[:pos] + piece + seq[pos:])

    return modified_sequences


def modify_lengths_by_adding_flanks(
    sequences: list[str],
    lengths: list[int],
    weights: list | np.ndarray,
    background_freqs: np.ndarray,
) -> list[str]:
    """Add random flanks to the input sequences.

    This function is intended for the simulation of MHC2 ligands of variable length.

    Args:
        sequences: Input sequences.
        lengths: The target lengths.
        weights: The weights/frequencies corresponding to the target lengths.
        background_freqs: The amino acid frequencies based on which the new sequence pieces are
            generated.

    Raises:
        ValueError: If not all input sequences have the same length.
        ValueError: If the smallest smallest target length is shorter that the input sequence
            length.

    Returns:
        The modified sequences.
    """
    # the length of the input sequences
    k = len(sequences[0])

    cum_background_freqs = np.cumsum(background_freqs)
    r = _draw_weighted(len(sequences), weights)

    modified_sequences = []
    for i, seq in enumerate(sequences):
        if len(seq) != k:
            raise ValueError("sequences do not have the same length")

        length = lengths[r[i]]

        if length == k:
            modified_sequences.append(seq)
        elif length < k:
            raise ValueError(
                f"smallest length {min(lengths)} is smaller than the input sequence length"
            )
        else:
            pos = np.random.randint(length - k + 1)
            piece = _generate_piece(length - k, cum_background_freqs)
            modified_sequences.append(piece[:pos] + seq + piece[pos:])

    return modified_sequences
