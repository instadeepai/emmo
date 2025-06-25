"""Module for computing various statistics and comparing distributions."""
from __future__ import annotations

import numpy as np


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
