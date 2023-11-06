"""Module for outputting matrices and responsibilities to a file."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from emmo.io.sequences import SequenceManager


def write_matrix(
    file_path: str | Path, matrix: np.ndarray, alphabet: str | tuple[str, ...] | list[str]
) -> None:
    """Write a PPM or PSSM to a file.

    Args:
        file_path: The file path.
        matrix: The matrix.
        alphabet: The alphabet.
    """
    file_path = Path(file_path)

    if len(matrix.shape) == 1:
        matrix = matrix[np.newaxis, :]

    df = pd.DataFrame(matrix, columns=list(alphabet))
    df.to_csv(file_path)


def write_matrices(
    directory: str | Path,
    matrices: np.ndarray,
    alphabet: str | tuple[str, ...] | list[str],
    flat_motif: np.ndarray | None = None,
    file_prefix: str = "",
) -> None:
    """Write multiple matrices into a directory.

    Args:
        directory: The directory.
        matrices: The matrices.
        alphabet: The alphabet
        flat_motif: An optional flat motif.
        file_prefix: The file prefix.
    """
    n = len(matrices)
    directory = Path(directory)

    for i, matrix in enumerate(matrices):
        write_matrix(directory / f"{file_prefix}matrix_{n}_{i+1}.csv", matrix, alphabet)

    if flat_motif is not None:
        write_matrix(directory / f"{file_prefix}matrix_{n}_flat.csv", flat_motif, alphabet)


def write_responsibilities(
    directory: str | Path,
    sequence_manager: SequenceManager,
    responsibilities: dict[int, np.ndarray],
    number_of_classes: int,
    file_prefix: str = "",
) -> None:
    """Write responsibilities into a directory.

    Args:
        directory: The directory.
        sequence_manager: The SequenceManager instance.
        responsibilities: The responsibility values.
        number_of_classes: The number of classes excluding the flat motif.
        file_prefix: The file prefix.
    """
    sm = sequence_manager
    directory = Path(directory)

    # collect the responsibilities
    all_responsibilities = np.zeros((sm.number_of_sequences(), number_of_classes + 1))

    for i, (length, s) in enumerate(sm.order_in_input_file):
        all_responsibilities[i, :] = responsibilities[length][s, :]

    df = pd.DataFrame(
        all_responsibilities,
        columns=[f"class{i+1}" for i in range(number_of_classes)] + ["flat_motif"],
    )

    df.insert(0, "peptide", sm.sequences)
    df["best_class"] = np.argmax(all_responsibilities, axis=1) + 1
    df.loc[(df["best_class"] == number_of_classes + 1), "best_class"] = "flat"

    df.to_csv(directory / f"{file_prefix}responsibilities.csv")
