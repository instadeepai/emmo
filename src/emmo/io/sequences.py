"""Module for handling input sequences."""
from __future__ import annotations

import collections
import itertools
from collections import Counter
from collections import defaultdict

import numpy as np

from emmo.constants import NATURAL_AAS
from emmo.io.file import load_csv
from emmo.io.file import load_txt
from emmo.io.file import Openable


def _read_sequences(file_path: Openable) -> list[str]:
    """Read sequences from a file.

    Args:
        file_path: The file path.

    Returns:
        The sequences.
    """
    return load_txt(file_path)


def _read_sequences_with_class_information(
    file_path: Openable, sequence_column: str, class_column: str
) -> tuple[list[str], list[str]]:
    """Read sequences with associated class.

    The input file is expected to have at least the column containing the sequences and one
    containing the classes.

    Args:
        file_path: The file path.
        sequence_column: Name of the column containing the sequences.
        class_column: Name of the column containing the associated classes.

    Returns:
        The sequences and their associated classes.
    """
    df = load_csv(file_path)
    seqs = df[sequence_column].astype(str).tolist()
    classes = df[class_column].astype(str).tolist()

    return seqs, classes


class SequenceManager:
    """Class for processing and organizing the input sequences."""

    def __init__(
        self,
        file_path: Openable,
        alphabet: str = "default",
        sequence_column: str | None = None,
        class_column: str | None = None,
    ) -> None:
        """Initialize the SequenceManager class.

        Args:
            file_path: The file path for the sequences.
            alphabet: The alphabet to use. Can be one of 'default' (natural amino acids), 'infer'
                (use characters appearing in the input sequences), or a string containing the
                characters explicitly.
            sequence_column: If the sequences come with associated classes, the name of the column
                with the sequences must be provided.
            class_column: If the sequences come with associated classes, the name of the column
                with the class information must be provided.

        Raises:
            ValueError: If the sequence_column argument is required but not provided.
        """
        if not class_column:
            self.sequences = _read_sequences(file_path)
            self.classes = None
        else:
            if not sequence_column:
                raise ValueError("please also provide 'sequence_column' argument")
            self.sequences, self.classes = _read_sequences_with_class_information(
                file_path, sequence_column, class_column
            )

        if alphabet == "default":
            self.alphabet = NATURAL_AAS
        elif alphabet == "infer":
            self.alphabet = tuple(sorted({a for seq in self.sequences for a in seq}))
        else:
            self.alphabet = tuple(sorted(alphabet))

        self.aa2idx = {a: i for i, a in enumerate(self.alphabet)}

        self._similarity_weights: dict[int, np.ndarray] = {}

    def number_of_sequences(self) -> int:
        """The total number of sequences.

        Returns:
            The total number of sequences.
        """
        return len(self.sequences)

    def get_minimal_length(self) -> int:
        """The minimal length among all sequences sequences.

        Returns:
            The minimal sequence length.
        """
        if not hasattr(self, "size_sorted_seqs"):
            self._sort_by_size()

        return self.min_length

    def get_maximal_length(self) -> int:
        """The maximal length among all sequences sequences.

        Returns:
            The maximal sequence length.
        """
        if not hasattr(self, "size_sorted_seqs"):
            self._sort_by_size()

        return self.max_length

    def sequences_as_indices(self) -> list[list[int]]:
        """The sequences encoded by the alphabet indices.

        Returns:
            The encoded sequences.
        """
        return [[self.aa2idx[a] for a in seq] for seq in self.sequences]

    def get_frequencies(self) -> np.ndarray:
        """The frequencies of the characters in the alphabet.

        Returns:
            The frequencies.
        """
        if not hasattr(self, "frequencies"):
            count_aas = Counter(itertools.chain.from_iterable(seq for seq in sequences))
            total = sum(count_aas.values())  # could be replaced by counter.total() in Python 3.10
            self.frequencies = np.array([count_aas[aa] for aa in self.alphabet]) / total

        return self.frequencies

    def get_size_sorted_sequences(self) -> dict[int, list[str]]:
        """The sequences sorted by their length into a dictionary.

        Returns:
            The keys are the sequence lengths and the values are lists of sequences of the
            respective length.
        """
        if not hasattr(self, "size_sorted_seqs"):
            self._sort_by_size()

        return self.size_sorted_seqs

    def get_size_sorted_arrays(self) -> dict[int, np.ndarray]:
        """The index-encoded sequences sorted by their length into a dictionary.

        Returns:
            The keys are the sequence lengths and the values are array of encoded sequences of the
            respective length.
        """
        if not hasattr(self, "size_sorted_arrays"):
            self._sort_by_size()

        return self.size_sorted_arrays

    def get_size_sorted_classes(self) -> dict[int, list[str]]:
        """The class information split by length of the sequences.

        Returns:
            The keys are the sequence lengths and the values are lists of class assignments in the
            order corresponding to the output of function 'get_size_sorted_sequences()'.
        """
        if not self.classes:
            raise RuntimeError("classes have not been set")

        if not hasattr(self, "size_sorted_seqs"):
            self._sort_by_size()

        return self.size_sorted_classes

    def _sort_by_size(self) -> None:
        """Initialize the sequences and encoded sequences sorted by their lengths."""
        self.size_sorted_seqs: dict[int, list[str]] = defaultdict(list)
        if self.classes:
            self.size_sorted_classes: dict[int, list[str]] = defaultdict(list)
        self.order_in_input_file: list[tuple[int, int]] = []

        for i, seq in enumerate(self.sequences):
            length = len(seq)

            self.order_in_input_file.append((length, len(self.size_sorted_seqs[length])))
            self.size_sorted_seqs[length].append(seq)
            if self.classes:
                self.size_sorted_classes[length].append(self.classes[i])

        self.size_sorted_arrays = {}

        for length, seqs in self.size_sorted_seqs.items():
            array = np.zeros((len(seqs), length), dtype=np.uint16)

            for i, seq in enumerate(seqs):
                for j, a in enumerate(seq):
                    array[i, j] = self.aa2idx[a]

            self.size_sorted_arrays[length] = array

        self.min_length = min(self.size_sorted_seqs.keys())
        self.max_length = max(self.size_sorted_seqs.keys())

    def get_similarity_weights(self, k: int = 9) -> np.ndarray:
        """The similarity weights for the sequences.

        The more k-mers a sequence shares with other sequences, the lower its weight.

        Args:
            k: The k-mer length use in the sequence weight calculation.

        Returns:
            The sequence weights for the specified k-mer length.
        """
        if k not in self._similarity_weights:
            self._similarity_weights[k] = similarity_weights(k, self.sequences)

        return self._similarity_weights[k]

    def split_array_by_size(self, array: np.ndarray) -> dict[int, np.ndarray]:
        """Split an array by the length of the input sequences.

        The input array must have the size (length of the first dimension) equal to the total
        number of sequences. These rows are then distributed over multiple new array (one for each
        available length) according to the length of the sequence at the the index.

        Args:
            array: The array to be split.

        Raises:
            ValueError: If the number of sequences and the size of the input array do not match.

        Returns:
            The array split by length of the sequences.
        """
        if len(array) != len(self.sequences):
            raise ValueError(
                "input must be an array with the same length as the list " "of sequences"
            )

        if not hasattr(self, "size_sorted_seqs"):
            self._sort_by_size()

        additional_dimensions = array.shape[1:]
        split_arrays = {
            length: np.zeros((len(seqs), *additional_dimensions), dtype=array.dtype)
            for length, seqs in self.size_sorted_seqs.items()
        }

        pos = dict.fromkeys(split_arrays, 0)

        for i, seq in enumerate(self.sequences):
            length = len(seq)
            split_arrays[length][pos[length]] = array[i]
            pos[length] += 1

        return split_arrays

    def recombine_split_array(self, split_array: dict[int, np.ndarray]) -> np.ndarray:
        """Recombine an array split by the sequence length.

        The resulting array is ordered according to the input sequence order (along the first
        dimension).

        Args:
            split_array: A dict with the sequence lengths as keys and numpy arrays as values.

        Returns:
            The recombined array.
        """
        recombined_array = None

        for i, (length, s) in enumerate(self.order_in_input_file):
            if recombined_array is None:
                recombined_array = np.zeros((len(self.sequences), *split_array[length].shape[1:]))

            recombined_array[i, :] = split_array[length][s, :]

        return recombined_array


def similarity_weights(k: int, sequences: list[str]) -> np.ndarray:
    """Calculate the similarity weights for the sequences.

    The more k-mers a sequence shares with other sequences, the lower its weight.

    Args:
        k: The k-mer length use in the sequence weight calculation.
        sequences: The sequences for which to calculate weights.

    Returns:
        The sequence weights for the specified k-mer length.
    """
    counts = count_k_mers(k, sequences)

    similarity_weights = np.ones(len(sequences), dtype=np.float64)

    for s, seq in enumerate(sequences):
        no_of_kmers = 0
        sum_of_counts = 0
        for i in range(len(seq) - k + 1):
            sum_of_counts += counts[seq[i : i + k]]
            no_of_kmers += 1
        if no_of_kmers:
            similarity_weights[s] = no_of_kmers / sum_of_counts

    return similarity_weights


def generate_k_mers(k: int, sequences: list[str]) -> collections.abc.Iterator[str]:
    """Generate all k-mers from a list of sequences.

    Args:
        k: The k-mer length.
        sequences: The list of sequences.

    Yields:
        The k-mers in order of appearance, can yield repeated k-mers.
    """
    for seq in sequences:
        for i in range(len(seq) - k + 1):
            yield seq[i : i + k]


def count_k_mers(k: int, sequences: list[str]) -> dict[str, int]:
    """Count the occurrences of all k-mers.

    Args:
        k: The k-mer length.
        sequences: The list of sequences.

    Returns:
        The number of occurrences for all k-mers appearing in the sequences.
    """
    return dict(Counter(generate_k_mers(k, sequences)))


if __name__ == "__main__":
    sequences = ["ABCDEFGHIJKLMNOPQ", "HIJKLMNOPQRST", "STUVPXYZ01", "abc"]
    counts = count_k_mers(9, sequences)
    print(counts)

    sw = similarity_weights(9, sequences)
    print(sw)
