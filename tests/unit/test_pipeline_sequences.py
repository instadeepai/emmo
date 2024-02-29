"""Test cases for the pipeline sequences functions."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emmo.constants import NATURAL_AAS
from emmo.io.file import save_txt
from emmo.pipeline.sequences import SequenceManager


@pytest.fixture()
def example_sequence_manager(example_sequences: list[str]) -> SequenceManager:
    """Sequence manager containing the example sequences (without class information)."""
    return SequenceManager(example_sequences, alphabet="default")


@pytest.fixture(scope="module")
def example_classes() -> list[str]:
    """Test classes."""
    return ["1", "2", "1", "3", "1", "1", "2", "2"]


@pytest.fixture(scope="module")
def txt_file_path(tmp_directory: Path, example_sequences: list[str]) -> Path:
    """TXT file path for test cases."""
    file_path = tmp_directory / "sequences.txt"

    save_txt(example_sequences, file_path)

    return file_path


@pytest.fixture(scope="module")
def csv_file_path(
    tmp_directory: Path, example_sequences: list[str], example_classes: list[str]
) -> Path:
    """CSV file path for test cases."""
    file_path = tmp_directory / "sequences.csv"

    df = pd.DataFrame({"peptide": example_sequences, "class": example_classes})
    df.to_csv(file_path, index=False, sep=",")

    return file_path


@pytest.fixture(scope="module")
def expected_indices() -> list[list[int]]:
    """Expected sequences returned by 'test_sequences_as_indices' for the example sequences."""
    return [
        [10, 0, 2, 15, 14, 2, 12, 0, 15, 2],
        [13, 10, 13, 6, 18, 8, 3, 13, 14, 0, 0, 13],
        [8, 0, 2, 17, 9, 16],
        [16, 5, 0, 5, 11, 12, 17, 5, 2, 8, 9, 11],
        [17, 7, 16, 17, 5, 12, 14, 5, 12, 9],
        [9, 17, 13, 2, 17, 17, 4, 16, 2, 3, 10, 0],
        [6, 4, 2, 14, 3, 14, 7, 12],
        [3, 14, 17, 17, 6, 0, 8, 5, 0, 5],
    ]


@pytest.fixture(scope="module")
def expected_size_sorted_sequences() -> dict[int, list[str]]:
    """Expected size-sorted sequences."""
    return {
        6: ["KADVLT"],
        8: ["HFDRERIP"],
        10: ["MADSRDPASD", "VITVGPRGPL", "ERVVHAKGAG"],
        12: ["QMQHWKEQRAAQ", "TGAGNPVGDKLN", "LVQDVVFTDEMA"],
    }


@pytest.fixture(scope="module")
def expected_size_sorted_arrays() -> dict[int, np.ndarray]:
    """Expected size-sorted sequences as indices."""
    sequences_as_indices = {
        6: [[8, 0, 2, 17, 9, 16]],
        8: [[6, 4, 2, 14, 3, 14, 7, 12]],
        10: [
            [10, 0, 2, 15, 14, 2, 12, 0, 15, 2],
            [17, 7, 16, 17, 5, 12, 14, 5, 12, 9],
            [3, 14, 17, 17, 6, 0, 8, 5, 0, 5],
        ],
        12: [
            [13, 10, 13, 6, 18, 8, 3, 13, 14, 0, 0, 13],
            [16, 5, 0, 5, 11, 12, 17, 5, 2, 8, 9, 11],
            [9, 17, 13, 2, 17, 17, 4, 16, 2, 3, 10, 0],
        ],
    }

    return {
        length: np.array(seqs, dtype=np.uint16) for length, seqs in sequences_as_indices.items()
    }


@pytest.fixture(scope="module")
def expected_size_sorted_classes() -> dict[int, list[str]]:
    """Expected size-sorted sequences."""
    return {
        6: ["1"],
        8: ["2"],
        10: ["1", "1", "2"],
        12: ["2", "3", "1"],
    }


@pytest.fixture(scope="module")
def example_array_to_split() -> np.ndarray:
    """An example array to be split according to the lengths of the example sequence."""
    return np.reshape(np.arange(80), (8, 10))


@pytest.fixture(scope="module")
def example_array_split() -> dict[int, np.ndarray]:
    """A split example array."""
    sequences_as_indices = {
        6: [np.arange(20, 30)],
        8: [np.arange(60, 70)],
        10: [
            np.arange(10),
            np.arange(40, 50),
            np.arange(70, 80),
        ],
        12: [
            np.arange(10, 20),
            np.arange(30, 40),
            np.arange(50, 60),
        ],
    }

    return {
        length: np.array(array, dtype=np.uint16) for length, array in sequences_as_indices.items()
    }


@pytest.fixture(scope="module")
def example_sequences_for_weights() -> list[str]:
    """Example sequences for similarity weight computation."""
    return [
        "MADSRDPASDQMQHWK",
        "MADSRDPASDQMQHWKEQRAA",
        "ASDQMQHWKEQRAAQKADVLTTGAGNPVGDKLN",
        "KEQRAAQKADVLTTGAGNPVGDKLN",
    ]


@pytest.mark.parametrize("alphabet", ["default", "".join(list(NATURAL_AAS) + ["B"])])
def test_initialization_with_unknown_amino_acids(
    alphabet: str, example_sequences: list[str]
) -> None:
    """Test that initialization of SequenceManager raises an error for unknown amino acids."""
    sequences = example_sequences + ["ACDXXXXX"]

    with pytest.raises(ValueError, match="the sequences contain letters that are not"):
        SequenceManager(sequences, alphabet=alphabet)


@pytest.mark.parametrize("alphabet", ["infer", "".join(list(NATURAL_AAS) + ["X"])])
def test_initialization_with_alternative_alphabet(
    alphabet: str, example_sequences: list[str]
) -> None:
    """Test that initialization of SequenceManager raises an error for unknown amino acids."""
    sequences = example_sequences + ["ACDXXXXX"]
    sequence_manager = SequenceManager(sequences, alphabet=alphabet)

    assert sequence_manager.sequences == sequences


def test_load_from_txt(txt_file_path: Path, example_sequences: list[str]) -> None:
    """Test that load sequences from a TXT file works as expected."""
    sequence_manager = SequenceManager.load_from_txt(txt_file_path)

    assert sequence_manager.sequences == example_sequences
    assert sequence_manager.classes is None
    assert sequence_manager.number_of_sequences() == len(example_sequences)
    assert sequence_manager.get_minimal_length() == min([len(seq) for seq in example_sequences])
    assert sequence_manager.get_maximal_length() == max([len(seq) for seq in example_sequences])


def test_load_from_csv(
    csv_file_path: Path,
    example_sequences: list[str],
    example_classes: list[str],
) -> None:
    """Test that loading a CSV file works as expected."""
    sequence_manager = SequenceManager.load_from_csv(csv_file_path, "peptide", class_column="class")

    assert sequence_manager.sequences == example_sequences
    assert sequence_manager.classes == example_classes
    assert len(sequence_manager.sequences) == len(sequence_manager.classes)
    assert sequence_manager.get_minimal_length() == min([len(seq) for seq in example_sequences])
    assert sequence_manager.get_maximal_length() == max([len(seq) for seq in example_sequences])


def test_load_from_csv_wo_classes(
    csv_file_path: Path,
    example_sequences: list[str],
) -> None:
    """Test that loading a CSV file without setting 'class_column' works as expected."""
    sequence_manager = SequenceManager.load_from_csv(csv_file_path, "peptide")

    assert sequence_manager.sequences == example_sequences
    assert sequence_manager.classes is None


def test_sequences_as_indices(
    example_sequence_manager: SequenceManager,
    expected_indices: list[list[int]],
) -> None:
    """Test that 'sequences_as_indices' works as expected."""
    assert example_sequence_manager.sequences_as_indices() == expected_indices


def test_get_frequencies(
    example_sequence_manager: SequenceManager, expected_amino_acid_frequencies: list[float]
) -> None:
    """Test that 'get_frequencies' works as expected."""
    assert np.allclose(example_sequence_manager.get_frequencies(), expected_amino_acid_frequencies)


def test_get_size_sorted_sequences(
    example_sequence_manager: SequenceManager, expected_size_sorted_sequences: dict[int, list[str]]
) -> None:
    """Test that 'get_size_sorted_sequences' works as expected."""
    assert example_sequence_manager.get_size_sorted_sequences() == expected_size_sorted_sequences


def test_get_size_sorted_arrays(
    example_sequence_manager: SequenceManager,
    expected_size_sorted_arrays: dict[int, np.ndarray],
) -> None:
    """Test that 'get_size_sorted_arrays' works as expected."""
    size_sorted_arrays = example_sequence_manager.get_size_sorted_arrays()

    assert set(size_sorted_arrays) == set(expected_size_sorted_arrays)
    for length in size_sorted_arrays:
        assert np.all(size_sorted_arrays[length] == expected_size_sorted_arrays[length])


def test_get_size_sorted_classes(
    example_sequences: list[str],
    example_classes: list[str],
    expected_size_sorted_classes: dict[int, list[str]],
) -> None:
    """Test that 'get_size_sorted_classes' works as expected."""
    sequence_manager = SequenceManager(example_sequences, classes=example_classes)

    assert sequence_manager.get_size_sorted_classes() == expected_size_sorted_classes


def test_split_array_by_size(
    example_sequence_manager: SequenceManager,
    example_array_to_split: np.ndarray,
    example_array_split: dict[int, np.ndarray],
) -> None:
    """Test that 'split_array_by_size' works as expected."""
    array_split = example_sequence_manager.split_array_by_size(example_array_to_split)

    assert set(array_split) == set(example_array_split)
    for length in array_split:
        assert np.all(array_split[length] == example_array_split[length])


def test_split_array_by_size_invalid_shape(example_sequence_manager: SequenceManager) -> None:
    """Test that 'split_array_by_size' throws a ValueError if the shape of the array is invalid.

    The first dimension of the array and the number of sequences must match.
    """
    invalid_length = example_sequence_manager.number_of_sequences() + 1
    array_to_split = np.reshape(np.arange(10 * invalid_length), (invalid_length, 10))

    with pytest.raises(
        ValueError, match="input must be an array with the same length as the list of sequences"
    ):
        example_sequence_manager.split_array_by_size(array_to_split)


def test_recombine_split_array(
    example_sequence_manager: SequenceManager,
    example_array_to_split: np.ndarray,
    example_array_split: dict[int, np.ndarray],
) -> None:
    """Test that 'recombine_split_array' works as expected."""
    recombined_array = example_sequence_manager.recombine_split_array(example_array_split)

    assert np.all(example_array_to_split == recombined_array)


@pytest.mark.parametrize(
    ("k", "expected_similarity_weights"),
    [
        (3, [0.4, 0.3877551, 0.42465753, 0.46]),
        (9, [0.47058824, 0.48148148, 0.51020408, 0.5]),
    ],
)
def test_get_similarity_weights(
    k: int,
    expected_similarity_weights: list[float],
    example_sequences_for_weights: list[str],
) -> None:
    """Test that 'get_similarity_weights' works as expected."""
    sequence_manager = SequenceManager(example_sequences_for_weights)
    similarity_weights = sequence_manager.get_similarity_weights(k=k)
    expected_similarity_weights_array = np.array(expected_similarity_weights, dtype=np.float64)

    assert similarity_weights.shape == expected_similarity_weights_array.shape
    assert np.allclose(similarity_weights, expected_similarity_weights_array)
