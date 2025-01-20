"""Module for allele name functions."""
from __future__ import annotations


GENES_MHC1 = ("A", "B", "C", "E", "G")
GENES_MHC2 = ("DR", "DP", "DQ")

MIN_NAME_LENGTH_MHC1 = 5
MAX_NAME_LENGTH_MHC1 = 6

# TODO: decide whether this should be set e.g. to 7 (DRA0101)
MIN_NAME_LENGTH_MHC2 = 3


def parse_mhc1_allele_pair(allele: str) -> str:
    """Parse an MHC1 allele pair in short format.

    The input must be a string in compact allele format.

    Args:
        allele: Allele.

    Raises:
        ValueError: If one of the alleles is too short or too long.
        ValueError: If the gene is unknown (not one of A, B, C, E, or G).
        ValueError: If the string could not be parsed.

    Returns:
        The input allele.
    """
    if len(allele) < MIN_NAME_LENGTH_MHC1:
        raise ValueError(f"allele '{allele}' is too short")

    if len(allele) > MAX_NAME_LENGTH_MHC1:
        raise ValueError(f"allele '{allele}' is too long")

    if allele[0] not in GENES_MHC1:
        raise ValueError(f"unknown gene '{allele[0]}', must be in {GENES_MHC1}")

    # check that the remaining characters are digits
    if not allele[1:].isdigit():
        raise ValueError(f"allele '{allele}' could not be parsed")

    return allele


def parse_mhc2_allele_pair(allele_pair: str) -> tuple[str, str]:
    """Parse an MHC2 allele pair.

    The input must be a string in which alpha and beta chain:
    - can be in arbitrary order
    - must be in compact allele format, separated by a hyphen ('-')
    - must have matching genes (one of DR, DP, or DQ)

    In case of DR, it is sufficient that the beta chain is provided in compact format.

    Args:
        allele_pair: Hyphen-separated allele pair (or DRB allele).

    Raises:
        ValueError: If one of the alleles is too short.
        ValueError: If only one allele is provided which is not DRB.
        ValueError: If the genes do not match for alpha and beta chain.
        ValueError: If the gene is unknown (not one of DR, DP, or DQ).
        ValueError: If alpha and beta chain cannot be assigned.
        ValueError: If the string could not be parsed.

    Returns:
        The alpha and beta chain in compact allele format.
    """
    alleles_split = allele_pair.split("-")

    # only beta chain of a DR allele is given
    if len(alleles_split) == 1:
        if len(allele_pair) < MIN_NAME_LENGTH_MHC2:
            raise ValueError(f"allele '{allele_pair}' is too short")

        if allele_pair[:3] != "DRB":
            raise ValueError("two alleles are needed except for locus DRB")

        return "DRA0101", allele_pair

    if len(alleles_split) != 2:
        raise ValueError(f"allele pair '{allele_pair}' could not be parsed")

    # alpha and beta chain are provided (in an arbitrary order)
    allele1, allele2 = alleles_split

    if len(allele1) <= MIN_NAME_LENGTH_MHC2:
        raise ValueError(f"allele '{allele1}' is too short")

    if len(allele2) <= MIN_NAME_LENGTH_MHC2:
        raise ValueError(f"allele '{allele2}' is too short")

    if allele1[:2] != allele2[:2]:
        raise ValueError(f"genes for the two alleles do not match in '{allele_pair}'")

    if allele1[:2] not in GENES_MHC2:
        raise ValueError(f"unknown gene '{allele1[:2]}', must be in {GENES_MHC2}")

    if allele1[2] == "A" and allele2[2] == "B":
        return allele1, allele2
    elif allele1[2] == "B" and allele2[2] == "A":
        return allele2, allele1
    else:
        raise ValueError(f"could not identify alpha and beta chain in '{allele_pair}'")
