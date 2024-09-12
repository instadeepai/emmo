"""Module for sequence distances."""
from __future__ import annotations

from typing import Iterable

import numpy as np

from emmo.constants import AA2IDX
from emmo.resources.allele2netmhc_pseudo_seq import sequences_mhc1
from emmo.resources.allele2netmhc_pseudo_seq import sequences_mhc2
from emmo.resources.substitution_matrix import BLOSUM62_MATRIX


def _mhc2_pseudosequence(allele: str) -> str:
    """Return the pseudosequence for an MHC2 alleles.

    The input allele must be the short name for the alpha and beta chain separated by a hyphen.

    Args:
        alleles: Iterable of allele names.

    Raises:
        KeyError: If an allele name was not found.

    Returns:
        The pseudosequences.
    """
    beta, alpha = allele.split("-")

    # swap if order was alpha-beta
    if "B" in alpha and "A" in beta:
        alpha, beta = beta, alpha

    # for now raise an exception if allele is not available
    if beta not in sequences_mhc2:
        raise KeyError(f"no pseudosequence found for beta chain {beta}")

    if alpha not in sequences_mhc2:
        raise KeyError(f"no pseudosequence found for alpha chain {alpha}")

    return sequences_mhc2[beta] + sequences_mhc2[alpha]


def _pseudosequences(
    alleles: Iterable[str], mhc_class: int | str, ignore_failed_lookups: bool = False
) -> dict[str, str]:
    """Return the pseudosequences for class I or II alleles.

    Args:
        alleles: iterable of allele names
        mhc_class: MHC class I or II
        ignore_failed_lookups: If True, skip alleles for which no pseudosequence is available, and
            otherwise raise an exception.

    Raises:
        ValueError: If the MHC class is not invalid.

    Returns:
        Allele names as keys and pseudosequences as values.
    """
    if mhc_class in (1, "1", "i", "I"):
        mhc_class = 1
    elif mhc_class in (2, "2", "ii", "II"):
        mhc_class = 2
    else:
        raise ValueError(f"unknown MHC class {mhc_class}")

    pseudosequences = {}

    for allele in alleles:
        try:
            if mhc_class == 1:
                seq = sequences_mhc1[allele]
            else:
                seq = _mhc2_pseudosequence(allele)
            pseudosequences[allele] = seq

        except KeyError as key_error:
            if not ignore_failed_lookups:
                raise key_error

    return pseudosequences


def get_blosum62_distance(
    sequence1: str, sequence2: str, gap_opening_cost: int = 11, gap_extension_cost: int = 1
) -> int:
    """Return the BLOSUM62 distance of two sequences of the same length.

    In case of gaps, this uses affine gap costs. The default values are chosen as the default
        values used by pblast.

    Args:
        sequence1: First amino acid sequence.
        sequence2: Second amino acid sequence.
        gap_opening_cost: Gap opening cost.
        gap_extension_cost: Gap extension cost.

    Returns:
        BLOSUM62 alignment score (with affine gap costs).
    """
    if len(sequence1) != len(sequence2):
        raise ValueError("sequences must have the same length")

    score = 0

    # remember status for affine gap costs
    gap_opened1, gap_opened2 = False, False

    for a, b in zip(sequence1, sequence2):
        # no gap
        if a != "-" and b != "-":
            # matrix is symmetric, so order does not matter
            score += BLOSUM62_MATRIX[AA2IDX[a], AA2IDX[b]]
            gap_opened1, gap_opened2 = False, False

        # only sequence 1 has gap
        elif a == "-" and b != "-":
            score -= gap_extension_cost if gap_opened1 else gap_opening_cost
            gap_opened1, gap_opened2 = True, False

        # only sequence 2 has gap
        elif a != "-" and b == "-":
            score -= gap_extension_cost if gap_opened2 else gap_opening_cost
            gap_opened1, gap_opened2 = False, True

        # both are gaps --> do nothing

    return score


def nearest_neighbors(
    available_alleles: list[str], unavailable_alleles: list[str], mhc_class: int | str
) -> dict[str, tuple[list[str], float]]:
    """Closest available alleles for a list of unavailable alleles.

    The distance of two alleles A and B is given by d = 1 - s(A,B) / (s(A,A) s(B,B)) ^ 0.5
    where s(A,B) is the BLOSUM62 alignment score.

    Args:
        available_alleles: Available alleles.
        unavailable_alleles: Alleles for which the closest available alleles shall be identified.
        mhc_class: MHC class I or II.

    Returns:
        A dict with the unavailable alleles as keys and as values a tuple of (i) the list of
        closest available alleles and (ii) the corresponding BLOSUM62-based distance.

    References:
        [1] M. Nielsen et al.
            Quantitative Predictions of Peptide Binding to Any HLA-DR Molecule of Known Sequence:
            NetMHCIIpan. PLoS Comput Biol. 2008 Jul; 4(7): e1000107.
            doi: 10.1371/journal.pcbi.1000107
    """
    mapping = {}

    pseudosequences = _pseudosequences(
        sorted(set(available_alleles) | set(unavailable_alleles)),
        mhc_class,
        ignore_failed_lookups=True,
    )

    for unavail_allele in unavailable_alleles:
        smallest_distance = float("inf")
        closest_alleles: list[str] = []

        if unavail_allele not in pseudosequences:
            mapping[unavail_allele] = (closest_alleles, smallest_distance)
            continue

        seq_a = pseudosequences[unavail_allele]

        for avail_allele in available_alleles:
            if avail_allele not in pseudosequences:
                continue

            seq_b = pseudosequences[avail_allele]

            s_ab = get_blosum62_distance(seq_a, seq_b)
            s_aa = get_blosum62_distance(seq_a, seq_a)
            s_bb = get_blosum62_distance(seq_b, seq_b)

            distance = 1 - s_ab / ((s_aa * s_bb) ** 0.5)

            if distance < smallest_distance:
                closest_alleles = [avail_allele]
                smallest_distance = distance
            elif distance == smallest_distance:
                closest_alleles.append(avail_allele)

        mapping[unavail_allele] = (closest_alleles, smallest_distance)

    return mapping


def pairwise_blosum62_scores(
    alleles: list[str],
    mhc_class: int | str,
    gap_opening_cost: int = 11,
    gap_extension_cost: int = 1,
) -> np.ndarray:
    """Compute the pairwise BLOSUM62 scores for a list of alleles.

    Args:
        alleles: List of alleles.
        mhc_class: MHC class I or II.
        gap_opening_cost: Gap opening cost.
        gap_extension_cost: Gap extension cost.

    Raises:
        TypeError: if the alleles input is not a list

    Returns:
        Pairwise BLOSUM62 scores.
    """
    # enforce list input since the resulting matrix will be constructed for the order in that list
    if not isinstance(alleles, list):
        raise TypeError("available_alleles must be of type list")

    pseudosequences = _pseudosequences(alleles, mhc_class, ignore_failed_lookups=False)

    n = len(alleles)
    matrix = np.zeros((n, n), dtype=np.int32)

    for i, allele1 in enumerate(alleles):
        for j, allele2 in enumerate(alleles):
            if i > j:
                continue

            score = get_blosum62_distance(
                pseudosequences[allele1],
                pseudosequences[allele2],
                gap_opening_cost=gap_opening_cost,
                gap_extension_cost=gap_extension_cost,
            )

            matrix[i, j] = score
            matrix[j, i] = score

    return matrix
