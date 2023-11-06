"""Module for handling the offset alignment."""
from __future__ import annotations


class AlignedOffsets:
    """Binding core offset alignment.

    This follows the strategy for binding core offset alignment in MoDec (Racle et al. 2019).
    Therein, the aligned offsets are annotated [-S, ..., -1, 0, 1, ..., S] where positive offsets
    mean that the core is shifted toward the C-terminus and negative offsets mean that the core is
    shifted towards the C-terminus, S is given by (max. sequence length - motif length + 1) // 2.
    The offset annotation 0 means that the core is exactly centered, which is only possible if the
    difference between the peptide and motif length is even.
    The offset annotations correspond to indices [0, ..., S * 2] in the offset dimension used in
    the EM algorithm / in the predictor. This class handles the mapping between the possible
    offsets for a given peptide (which can be shorter than the max. sequence length) and the
    indices [0, ..., S * 2] according to the offset annotation.
    """

    def __init__(
        self,
        motif_length: int,
        max_sequence_length: int,
        force_continuous_offset_lists: bool = False,
    ) -> None:
        """Initialize the AlignedOffstes class.

        In MoDec, the offset position annotated as "0" is skipped for all sequences where
        (seq. length - motif length) is odd, here a continuous offset list
        (i.e. [..., -1, 0, 1, ...]) can optionally be used.

        Args:
            motif_length: The motif length.
            max_sequence_length: The maximal sequence length.
            force_continuous_offset_lists: Whether to use the continuous offsets even if the
                difference between motif and peptide length is odd.
        """
        self.motif_length = motif_length
        self.max_sequence_length = max_sequence_length

        # maximal binding core offset (symmetric around 0 as in MoDec)
        self.max_offset = (max_sequence_length - motif_length + 1) // 2
        self.n_offsets = 2 * self.max_offset + 1
        self.force_continuous_offset_lists = force_continuous_offset_lists

        # maps the length of a sequence to the list of relevant offsets
        self._offset_lists: dict[int, list[int]] = {}

    def get_number_of_offsets(self) -> int:
        """Total number of available offsets.

        This is the number of offset in the maximal sequence length (possible plus 1 since the
        offset annotations are symmetric around 0).

        Returns:
            Number of offsets.
        """
        return self.n_offsets

    def get_offset_list(self, length: int) -> list[int]:
        """Offset indices for a specific peptide length.

        Args:
            length: Peptide length.

        Returns:
            List of offset indices.
        """
        if length not in self._offset_lists:
            self._offset_list_for_single_length(length)

        return self._offset_lists[length]

    def _offset_list_for_single_length(self, length: int) -> None:
        """Calculate offset list for a specific peptide length.

        Args:
            length: List of offset indices.
        """
        # maximal offset for the length handled here
        s_max = (length - self.motif_length + 1) // 2

        # global maximal offset
        k = (self.n_offsets - 1) // 2

        if (length - self.motif_length) % 2 == 0:
            offset_list = [s + k for s in range(-s_max, s_max + 1)]
        elif self.force_continuous_offset_lists:
            offset_list = [s + k for s in range(-s_max, s_max)]
        else:
            # skip offset "0"
            offset_list = [s + k for s in range(-s_max, 0)]
            offset_list.extend([s + k for s in range(1, s_max + 1)])

        self._offset_lists[length] = offset_list

    def get_offset_annotation(self) -> list[int]:
        """Offset annotations.

        Returns a list [-S, ..., -1, 0, 1, ..., S] where  S is given by
        (max. sequence length - motif length + 1) // 2.

        Returns:
            Offset annotations.
        """
        return list(range(-self.max_offset, self.max_offset + 1))

    def get_offset_in_sequence(
        self,
        sequence_length: int,
        offset_in_matrix: int,
    ) -> int:
        """Map an offset index to the correspond offset index for the specified sequence length.

        Args:
            sequence_length: Sequence length.
            offset_in_matrix: Offset to be mapped.

        Returns:
            The corresponding offset index for the given sequence length.
        """
        return self.get_offset_list(sequence_length).index(offset_in_matrix)
