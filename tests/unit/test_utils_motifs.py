"""Test cases for the utils motifs functions."""
from __future__ import annotations

import numpy as np
import pytest

from emmo.pipeline.background import Background
from emmo.utils.motifs import _target_frequencies
from emmo.utils.motifs import count_amino_acids
from emmo.utils.motifs import position_probability_matrix
from emmo.utils.motifs import pseudocount_frequencies
from emmo.utils.motifs import total_frequencies

ERROR_MSG_NO_AMINO_ACIDS = "sum of all amino acid counts must be greater than zero"


@pytest.fixture()
def expected_position_probability_matrix() -> list[list[float]]:
    """Expected position probability matrix.

    This returns the position probability matrix for the first two amino acid positions in the
    equal-length example sequences.
    """
    return [
        [
            0.25,
            0.0,
            0.0,
            0.125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.125,
            0.25,
            0.0,
            0.0,
            0.125,
            0.0,
            0.0,
            0.0,
            0.125,
            0.0,
            0.0,
        ],
        [
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.125,
            0.0,
            0.125,
            0.0,
            0.0,
            0.125,
            0.0,
            0.0,
            0.125,
            0.125,
            0.0,
            0.0,
            0.125,
            0.0,
            0.0,
        ],
    ]


def test_target_frequencies() -> None:
    """Ensure that values in '_target_frequencies()' sum up to 1."""
    assert np.isclose(np.sum(_target_frequencies()), 1.0)


def test_pseudocount_frequencies() -> None:
    """Ensure that 'pseudocount_frequencies' returns only non-zero values."""
    example_freqs = Background("uniprot").frequencies.copy()

    example_freqs[1] = 0
    example_freqs[3] = 0
    example_freqs[10] = 0

    pseudocount_freqs = pseudocount_frequencies(example_freqs)

    assert example_freqs.shape == pseudocount_freqs.shape
    assert not (example_freqs > 0.0).all()
    assert (pseudocount_freqs > 0.0).all()


def test_count_amino_acids(
    example_sequences: list[str],
    expected_amino_acid_counts: list[int],
) -> None:
    """Test the 'count_amino_acids' function."""
    assert count_amino_acids(example_sequences) == expected_amino_acid_counts
    assert count_amino_acids("".join(example_sequences)) == expected_amino_acid_counts


def test_count_amino_acids_with_invalid_amino_acid() -> None:
    """Ensure that 'count_amino_acids' raises ValueError for unknown amino acids."""
    sequences = "AXAAAAAA"

    with pytest.raises(ValueError, match="found unknown amino acid"):
        count_amino_acids(sequences)


def test_total_frequencies(
    example_sequences: list[str],
    expected_amino_acid_frequencies: list[float],
) -> None:
    """Test the 'total_frequencies' function."""
    assert np.allclose(total_frequencies(example_sequences), expected_amino_acid_frequencies)
    assert np.allclose(
        total_frequencies("".join(example_sequences)), expected_amino_acid_frequencies
    )


@pytest.mark.parametrize(
    ("sequences", "match"),
    [
        ("AXAAAAAA", "found unknown amino acid"),
        ("", ERROR_MSG_NO_AMINO_ACIDS),
        ([""], ERROR_MSG_NO_AMINO_ACIDS),
    ],
)
def test_total_frequencies_with_invalid_input(sequences: str | list[str], match: str) -> None:
    """Ensure that 'total_frequencies' raises ValueError for invalid inputs.

    Invalid inputs are sequences that contain unknown amino acids or if the sum of counts is zero,
    i.e., if only empty strings have been provided.
    """
    with pytest.raises(ValueError, match=match):
        total_frequencies(sequences)


def test_total_position_probability_matrix(
    example_sequences_equal_length: list[str],
    expected_position_probability_matrix: list[list[float]],
) -> None:
    """Test the 'position_probability_matrix' function."""
    assert np.allclose(
        position_probability_matrix(
            [seq[:2] for seq in example_sequences_equal_length], use_pseudocounts=False
        ),
        expected_position_probability_matrix,
    )
