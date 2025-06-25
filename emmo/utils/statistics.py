"""Module for computing various statistics and comparing distributions."""
from __future__ import annotations

import numpy as np

from emmo.constants import EPSILON
from emmo.utils import logger

log = logger.get(__name__)


def compute_aic(number_of_parameters: int, log_likelihood: float) -> float:
    """Akaike information criterion.

    Args:
        number_of_parameters: Number of parameters.
        log_likelihood: Log likelihood.

    Returns:
        Akaike information criterion.
    """
    return 2 * number_of_parameters - 2 * log_likelihood


def kullback_leibler_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Kullback-Leibler (KL) divergence.

    Args:
        a: Distribution a.
        b: Distribution b.

    Raises:
        ValueError: If the two distribution arrays do not have the same shape.

    Returns:
        KL distance of the two distributions.
    """
    if a.shape != b.shape:
        raise ValueError("input arrays must have the same shape")

    a = _ensure_positive_values(a)
    b = _ensure_positive_values(b)

    return np.sum(a * np.log2(a / b))


def symmetrized_kullback_leibler_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Symmetrized Kullback-Leibler divergence between two distributions p and q.

    This is defined as: D_KL(p || q) + D_KL(q || p) where D_KL is the Kullback-Leibler divergence.

    Args:
        p: First probability distribution.
        q: Second probability distribution.

    Returns:
        Symmetrized Kullback-Leibler divergence.
    """
    p = _ensure_positive_values(p)
    q = _ensure_positive_values(q)

    return kullback_leibler_divergence(p, q) + kullback_leibler_divergence(q, p)


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Average euclidean distance of two arrays.

    Args:
        a: Second array.
        b: First array.

    Raises:
        ValueError: If the two arrays do not have the same shape.

    Returns:
        Average euclidean distance of two arrays.
    """
    if a.shape != b.shape:
        raise ValueError("input arrays must have the same shape")

    length = a.shape[0]

    return (1 / length) * np.sum(np.square(a - b))


def _ensure_positive_values(arr: np.ndarray) -> np.ndarray:
    """Ensure that the input array has only positive values.

    The function throws a ValueError if any value in the array is negative. Zero values are replaced
    with a small positive constant to avoid issues in calculations.

    Args:
        arr: Input array to check.

    Returns:
        The input array with zero values replaced by a small positive constant.

    Raises:
        ValueError: If the input array contains negative values.
    """
    if np.any(arr < 0):
        raise ValueError("array must be a non-negative distribution")

    if np.any(arr == 0):
        log.warning(f"array contains zero values, replacing with {EPSILON}")
        arr = np.where(arr == 0, EPSILON, arr)

    return arr
