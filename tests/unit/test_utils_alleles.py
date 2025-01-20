"""Test cases for the utils alleles functions."""
from __future__ import annotations

import pytest

from emmo.utils.alleles import parse_mhc2_allele_pair


@pytest.mark.parametrize(
    ("allele_pair", "expected_return_value"),
    [
        # first alpha, second beta
        ("DRA0101-DRB10102", ("DRA0101", "DRB10102")),
        ("DPA10202-DPB10101", ("DPA10202", "DPB10101")),
        ("DQA10201-DQB10201", ("DQA10201", "DQB10201")),
        # first beta, second alpha
        ("DRB10102-DRA0101", ("DRA0101", "DRB10102")),
        ("DPB10101-DPA10202", ("DPA10202", "DPB10101")),
        ("DQB10201-DQA10201", ("DQA10201", "DQB10201")),
        # only beta
        ("DRB10102", ("DRA0101", "DRB10102")),
    ],
)
def test_parse_mhc2_allele_pair(allele_pair: str, expected_return_value: tuple[str, str]) -> None:
    """Test that 'parse_mhc2_allele_pair' works as expected.

    Args:
        allele_pair: Input string test.
        expected_return_value: Expected output of the function.
    """
    assert parse_mhc2_allele_pair(allele_pair) == expected_return_value


@pytest.mark.parametrize(
    ("allele_pair", "expected_error_message"),
    [
        ("DR", "allele 'DR' is too short"),
        ("DRA0101-DR", "allele 'DR' is too short"),
        ("DP-DPB10101", "allele 'DP' is too short"),
        ("DPB10101", "two alleles are needed except for locus DRB"),
        ("DRA0101-DPA10202", "genes for the two alleles do not match in 'DRA0101-DPA10202'"),
        ("DPA10202-DPA10101", "could not identify alpha and beta chain in 'DPA10202-DPA10101'"),
        ("DXA10201-DXB10201", "unknown gene 'DX', must be in"),
        (
            "DRA0101-DRB10102-DRB10102",
            "allele pair 'DRA0101-DRB10102-DRB10102' could not be parsed",
        ),
    ],
)
def test_parse_mhc2_allele_pair_invalid_input(
    allele_pair: str, expected_error_message: tuple[str, str]
) -> None:
    """Ensure that 'parse_mhc2_allele_pair' raises a ValueError for an invalid input.

    Args:
        allele_pair: Input string test.
        expected_error_message: Expected error message match.
    """
    with pytest.raises(ValueError, match=expected_error_message):
        parse_mhc2_allele_pair(allele_pair)
