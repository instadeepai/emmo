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


class SequenceManager:
    """Class for processing and organizing the input sequences."""

    def __init__(
        self,
        sequences: list[str],
        alphabet: str = "default",
        classes: list[str] | None = None,
    ) -> None:
        """Initialize the SequenceManager class.

        Args:
            sequences: The sequences.
            alphabet: The alphabet to use. Can be one of 'default' (natural amino acids), 'infer'
                (use characters appearing in the input sequences), or a string containing the
                characters explicitly.
            classes: A list containing the classes to which the each individual sequence belongs.

        Raises:
            ValueError: If 'classes' are provided but the number of sequences and the length of the
                'classes' list differ.
        """
        self.sequences = sequences
        self.classes = classes

        if self.classes is not None:
            if len(self.sequences) != len(self.classes):
                raise ValueError("the sequence and classes lists must have the same length")

        letters_in_sequences = {a for seq in self.sequences for a in seq}

        if alphabet == "default":
            self.alphabet = NATURAL_AAS
            if not letters_in_sequences.issubset(set(NATURAL_AAS)):
                raise ValueError(
                    "the sequences contain letters that are not standard amino acids, use "
                    "alphabet='infer' or provide a custom alphabet"
                )
        elif alphabet == "infer":
            self.alphabet = tuple(sorted(letters_in_sequences))
        else:
            self.alphabet = tuple(sorted(alphabet))
            if not letters_in_sequences.issubset(set(self.alphabet)):
                raise ValueError(
                    "the sequences contain letters that are not in the provided alphabet"
                )

        self.aa2idx = {a: i for i, a in enumerate(self.alphabet)}

        self._similarity_weights: dict[int, np.ndarray] = {}

    @property
    def number_of_sequences(self) -> int:
        """The total number of sequences."""
        return len(self.sequences)

    @property
    def size_sorted_sequences(self) -> dict[int, list[str]]:
        """Dict mapping the sequence lengths to the corresponding sublists of sequences."""
        if not hasattr(self, "_size_sorted_sequences"):
            self._size_sorted_sequences: dict[int, list[str]] = defaultdict(list)

            for seq in self.sequences:
                self._size_sorted_sequences[len(seq)].append(seq)

        return self._size_sorted_sequences

    @property
    def min_length(self) -> int:
        """The minimal length among all sequences sequences."""
        return min(self.size_sorted_sequences.keys())

    @property
    def max_length(self) -> int:
        """The maximal length among all sequences sequences."""
        return max(self.size_sorted_sequences.keys())

    @property
    def size_sorted_classes(self) -> dict[int, list[str]]:
        """Dict mapping the sequence lengths to the corresponding sublists of class annotations.

        Raises:
            RuntimeError: If no class information is available.
        """
        if not self.classes:
            raise RuntimeError("classes have not been set")

        if not hasattr(self, "_size_sorted_classes"):
            self._size_sorted_classes: dict[int, list[str]] = defaultdict(list)

            for i, seq in enumerate(self.sequences):
                self.size_sorted_classes[len(seq)].append(self.classes[i])

        return self._size_sorted_classes

    @property
    def order_in_input_file(self) -> list[tuple[int, int]]:
        """List for recombining arrays that have been split by the lengths of the sequences.

        A list of n tuples of the form (length_i, idx_in_sublist_i) where n is the total
        number of sequences, length_i is the length of sequence i and idx_in_sublist_i is the
        index of the item belonging to sequence i in the sublist corresponding to length_i.
        """
        if not hasattr(self, "_order_in_input_file"):
            self._order_in_input_file: list[tuple[int, int]] = []
            counter: dict[int, int] = defaultdict(int)

            for seq in self.sequences:
                length = len(seq)
                self._order_in_input_file.append((length, counter[length]))
                counter[length] += 1

        return self._order_in_input_file

    @property
    def size_sorted_arrays(self) -> dict[int, np.ndarray]:
        """Dict mapping the sequence lengths to the index-encoded arrays of sequences."""
        if not hasattr(self, "_size_sorted_arrays"):
            self._size_sorted_arrays: dict[int, np.ndarray] = defaultdict(list)

            for length, seqs in self.size_sorted_sequences.items():
                array = np.zeros((len(seqs), length), dtype=np.uint16)

                for i, seq in enumerate(seqs):
                    for j, a in enumerate(seq):
                        array[i, j] = self.aa2idx[a]

                self._size_sorted_arrays[length] = array

        return self._size_sorted_arrays

    @property
    def frequencies(self) -> np.ndarray:
        """The frequencies of the characters in the alphabet."""
        if not hasattr(self, "_frequencies"):
            count_aas = Counter(itertools.chain.from_iterable(seq for seq in self.sequences))
            total = sum(count_aas.values())  # could be replaced by counter.total() in Python 3.10
            self._frequencies = np.array([count_aas[aa] for aa in self.alphabet]) / total

        return self._frequencies

    @classmethod
    def load_from_txt(cls, file_path: Openable, alphabet: str = "default") -> SequenceManager:
        """Read sequences from a text file.

        Args:
            file_path: The file path.
            alphabet: The alphabet to use. Can be one of 'default' (natural amino acids), 'infer'
                (use characters appearing in the input sequences), or a string containing the
                characters explicitly.

        Returns:
            A sequence manager containing the loaded sequences.
        """
        sequences = load_txt(file_path)

        return cls(sequences, alphabet=alphabet, classes=None)

    @classmethod
    def load_from_csv(
        cls,
        file_path: Openable,
        sequence_column: str,
        class_column: str | None = None,
        alphabet: str = "default",
    ) -> SequenceManager:
        """Read sequences from a csv file, optianally with associated class information.

        Args:
            file_path: The path to the csv file.
            sequence_column: The name of the column containing the sequences.
            class_column: The name of the column containing the class information.
            alphabet: The alphabet to use. Can be one of 'default' (natural amino acids), 'infer'
                (use characters appearing in the input sequences), or a string containing the
                characters explicitly.

        Returns:
            A sequence manager containing the loaded sequences and optianally the class information.
        """
        df = load_csv(file_path)
        sequences = df[sequence_column].astype(str).tolist()
        classes = df[class_column].astype(str).tolist() if class_column is not None else None

        return cls(sequences, alphabet=alphabet, classes=classes)

    def sequences_as_indices(self) -> list[list[int]]:
        """The sequences encoded by the alphabet indices.

        Returns:
            The encoded sequences.
        """
        return [[self.aa2idx[a] for a in seq] for seq in self.sequences]

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
            raise ValueError("input must be an array with the same length as the list of sequences")

        additional_dimensions = array.shape[1:]
        split_arrays = {
            length: np.zeros((len(seqs), *additional_dimensions), dtype=array.dtype)
            for length, seqs in self.size_sorted_sequences.items()
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
