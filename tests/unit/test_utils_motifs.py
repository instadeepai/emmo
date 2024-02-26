"""Test cases for the utils motifs functions."""
from __future__ import annotations

import numpy as np
import pytest

from emmo.resources.background_freqs import get_background
from emmo.utils.motifs import _target_frequencies
from emmo.utils.motifs import count_amino_acids
from emmo.utils.motifs import position_probability_matrix
from emmo.utils.motifs import pseudocount_frequencies
from emmo.utils.motifs import total_frequencies


EXAMPLE_SEQUENCES: list[str] = [
    "MADSRDPASD",
    "QMQHWKEQRA",
    "AQKADVLTTG",
    "AGNPVGDKLN",
    "VITVGPRGPL",
    "LVQDVVFTDE",
    "MAHFDRERIP",
    "ERVVHAKGAG",
]


@pytest.fixture(scope="module")
def expected_amino_acid_counts() -> list[int]:
    """Expected amino acid counts."""
    return [9, 0, 8, 4, 2, 7, 3, 2, 4, 4, 3, 2, 5, 5, 6, 2, 4, 9, 1, 0]


@pytest.fixture(scope="module")
def expected_amino_acid_frequencies() -> list[float]:
    """Expected amino acid frequencies."""
    return [
        0.1125,
        0.0,
        0.1,
        0.05,
        0.025,
        0.0875,
        0.0375,
        0.025,
        0.05,
        0.05,
        0.0375,
        0.025,
        0.0625,
        0.0625,
        0.075,
        0.025,
        0.05,
        0.1125,
        0.0125,
        0.0,
    ]


@pytest.fixture()
def expected_position_probability_matrix() -> list[list[float]]:
    """Expected position probability matrix."""
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
        [
            0.0,
            0.0,
            0.125,
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
            0.25,
            0.0,
            0.0,
            0.125,
            0.125,
            0.0,
            0.0,
        ],
        [
            0.125,
            0.0,
            0.125,
            0.0,
            0.125,
            0.0,
            0.125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.125,
            0.0,
            0.0,
            0.125,
            0.0,
            0.25,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
            0.125,
            0.125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.125,
            0.0,
            0.0,
            0.25,
            0.125,
            0.0,
        ],
        [
            0.125,
            0.0,
            0.125,
            0.0,
            0.0,
            0.125,
            0.0,
            0.0,
            0.125,
            0.0,
            0.0,
            0.0,
            0.125,
            0.0,
            0.125,
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
        ],
        [
            0.0,
            0.0,
            0.125,
            0.25,
            0.125,
            0.0,
            0.0,
            0.0,
            0.125,
            0.125,
            0.0,
            0.0,
            0.125,
            0.0,
            0.125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
            0.125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.125,
            0.125,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.125,
            0.0,
            0.125,
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
            0.125,
            0.125,
            0.125,
            0.0,
            0.0,
            0.0,
        ],
        [
            0.125,
            0.0,
            0.125,
            0.125,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            0.125,
            0.0,
            0.125,
            0.125,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
    ]


def test_target_frequencies() -> None:
    """Ensure that values in '_target_frequencies()' sum up to 1."""
    assert np.isclose(np.sum(_target_frequencies()), 1.0)


def test_pseudocount_frequencies() -> None:
    """Ensure that pseudocount_frequencies returns only non-zero values."""
    example_freqs = get_background("MHC1_biondeep")

    example_freqs[1] = 0
    example_freqs[3] = 0
    example_freqs[10] = 0

    pseudocount_freqs = pseudocount_frequencies(example_freqs)

    assert example_freqs.shape == pseudocount_freqs.shape
    assert not (example_freqs > 0.0).all()
    assert (pseudocount_freqs > 0.0).all()


@pytest.mark.parametrize("sequences", [EXAMPLE_SEQUENCES, "".join(EXAMPLE_SEQUENCES)])
def test_count_amino_acids(
    sequences: str | list[str], expected_amino_acid_counts: list[int]
) -> None:
    """Test the 'expected_amino_acid_counts' function."""
    assert count_amino_acids(sequences) == expected_amino_acid_counts


def test_count_amino_acids_with_invalid_amino_acid() -> None:
    """Ensure that 'expected_amino_acid_counts' raises ValueError for unknown amino acids."""
    sequences = "AXAAAAAA"

    with pytest.raises(ValueError, match="found unknown amino acid"):
        count_amino_acids(sequences)


@pytest.mark.parametrize("sequences", [EXAMPLE_SEQUENCES, "".join(EXAMPLE_SEQUENCES)])
def test_total_frequencies(
    sequences: str | list[str], expected_amino_acid_frequencies: list[float]
) -> None:
    """Test the 'expected_amino_acid_counts' function."""
    assert np.allclose(total_frequencies(sequences), expected_amino_acid_frequencies)


@pytest.mark.parametrize(
    ("sequences", "match"),
    [
        ("AXAAAAAA", "found unknown amino acid"),
        ("", "sum of all amino acid counts must be greater than zero"),
        ([""], "sum of all amino acid counts must be greater than zero"),
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
    expected_position_probability_matrix: list[list[float]],
) -> None:
    """Test the 'position_probability_matrix' function."""
    assert np.allclose(
        position_probability_matrix(EXAMPLE_SEQUENCES, use_pseudocounts=False),
        expected_position_probability_matrix,
    )
