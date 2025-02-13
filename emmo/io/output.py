"""Module for outputting matrices and responsibilities to a file."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from emmo.io.file import Openable
from emmo.io.file import save_csv
from emmo.pipeline.sequences import SequenceManager
from emmo.utils import logger

log = logger.get(__name__)


SKIP_DIRECTORIES = ["plots"]


def check_valid_directory_name(directory_name: str) -> None:
    """Check if a directory name is valid for Linux and Windows.

    Args:
        directory_name: The directory name.

    Raises:
        ValueError: If the directory name is not valid.
    """
    # must not be empty
    if not directory_name:
        raise ValueError("directory name must not be empty")

    # not any of the following characters: < > : " / \ | ? *
    if re.search(r"[<>:\"/\\|?*]", directory_name):
        raise ValueError(
            'directory name must not contain any of the following characters: < > : " / \\ | ? *'
        )

    # no reserved strings
    if directory_name in (
        ".",
        "..",
        "CON",
        "PRN",
        "AUX",
        "NUL",
        "COM1",
        "COM2",
        "COM3",
        "COM4",
        "COM5",
        "COM6",
        "COM7",
        "COM8",
        "COM9",
        "LPT1",
        "LPT2",
        "LPT3",
        "LPT4",
        "LPT5",
        "LPT6",
        "LPT7",
        "LPT8",
        "LPT9",
    ):
        raise ValueError(f"directory name '{directory_name}' is a reserved string")

    # not ASCII control characters
    if any(ord(c) < 32 for c in directory_name):
        raise ValueError("directory name must not contain ASCII control characters")

    # must not begin or end with a space
    if directory_name[0] == " " or directory_name[-1] == " ":
        raise ValueError("directory name must not begin or end with a space")


def write_matrix(
    file_path: Openable,
    matrix: np.ndarray,
    alphabet: str | tuple[str, ...] | list[str],
    force: bool = False,
) -> None:
    """Write a PPM or PSSM to a file.

    Args:
        file_path: The file path.
        matrix: The matrix.
        alphabet: The alphabet.
        force: Overwrite the file if it already exists.
    """
    if len(matrix.shape) == 1:
        matrix = matrix[np.newaxis, :]

    df = pd.DataFrame(matrix, columns=list(alphabet))
    save_csv(df, file_path, force=force)


def write_matrices(
    directory: Openable,
    matrices: np.ndarray,
    alphabet: str | tuple[str, ...] | list[str],
    flat_motif: np.ndarray | None = None,
    file_prefix: str = "",
    force: bool = False,
) -> None:
    """Write multiple matrices into a directory.

    Args:
        directory: The directory.
        matrices: The matrices.
        alphabet: The alphabet
        flat_motif: An optional flat motif.
        file_prefix: The file prefix.
        force: Overwrite files if they already exist.
    """
    n = len(matrices)
    directory = AnyPath(directory)

    for i, matrix in enumerate(matrices):
        write_matrix(
            directory / f"{file_prefix}matrix_{n}_{i+1}.csv", matrix, alphabet, force=force
        )

    if flat_motif is not None:
        write_matrix(
            directory / f"{file_prefix}matrix_{n}_flat.csv", flat_motif, alphabet, force=force
        )


def write_responsibilities(
    directory: Openable,
    sequence_manager: SequenceManager,
    responsibilities: dict[int, np.ndarray],
    number_of_classes: int,
    file_prefix: str = "",
    force: bool = False,
) -> None:
    """Write responsibilities into a directory.

    Args:
        directory: The directory.
        sequence_manager: The SequenceManager instance.
        responsibilities: The responsibility values.
        number_of_classes: The number of classes excluding the flat motif.
        file_prefix: The file prefix.
        force: Overwrite the file if it already exists.
    """
    directory = AnyPath(directory)

    # collect the responsibilities
    all_responsibilities = np.zeros((sequence_manager.number_of_sequences, number_of_classes + 1))

    for i, (length, s) in enumerate(sequence_manager.order_in_input_file):
        all_responsibilities[i, :] = responsibilities[length][s, :]

    df = pd.DataFrame(
        all_responsibilities,
        columns=[f"class{i+1}" for i in range(number_of_classes)] + ["flat_motif"],
    )

    df.insert(0, "peptide", sequence_manager.sequences)
    df["best_class"] = np.argmax(all_responsibilities, axis=1) + 1
    df.loc[(df["best_class"] == number_of_classes + 1), "best_class"] = "flat"

    save_csv(df, directory / f"{file_prefix}responsibilities.csv", force=force)


def find_deconvolution_results(directory: Openable) -> pd.DataFrame:  # noqa: CCR001
    """Find models in the output directory of a per-group deconvolution.

    Args:
        directory: The output directory of the per-group deconvolution.

    Raises:
        ValueError: If no models were found.

    Returns:
        A DataFrame containing columns:
            - 'group': The group (i.e. the corresponding subdirectory name).
            - 'number_of_classes': Number of classes used in the deconvolution.
            - 'model_path': The path to the model directory.
    """
    directory = AnyPath(directory)

    data = []

    for group_dir in directory.iterdir():
        if not group_dir.is_dir() or group_dir.name in SKIP_DIRECTORIES:
            continue

        group = group_dir.name

        for model_dir in group_dir.iterdir():
            if not model_dir.is_dir() or model_dir.name in SKIP_DIRECTORIES:
                continue

            number_of_classes = _get_number_of_classes(model_dir)
            if number_of_classes is None:
                continue

            data.append(
                {
                    "group": group,
                    "number_of_classes": number_of_classes,
                    "model_path": model_dir,
                }
            )

    if not data:
        raise ValueError(f"no model directories found in input directory '{directory}'")

    return pd.DataFrame(data).sort_values(by=["group", "number_of_classes"], ignore_index=True)


def _get_number_of_classes(model_dir: AnyPath) -> int | None:
    """Get the number of classes from the directory name.

    Args:
        model_dir: Directory path.

    Returns:
        The number of classes parsed from the directory name.
    """
    # backward compatibility: the current naming convention is 'classes_{num_of_classes}',
    # the old naming convention is 'clusters{num_of_classes}'
    match = re.match(r"((classes_)|(clusters))([1-9]\d*)", model_dir.name)
    if not match:
        log.warning(
            f"num. of classes could not be parsed from directory '{model_dir}', it will be skipped"
        )
        return None

    return int(match.group(4))
