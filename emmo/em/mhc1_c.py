"""Expectation-maximization-based deconvolution of MHC1 ligands.

This mostly follows the methods in Gfeller et al. 2018 with the associated tool
MixMHCp. Modifications are:
- The sequences of length other than the motif length are also used in the
  estimation of the PPMs whereas MixMHCp only uses the sequences of the same
  length (usually 9-mers), then fixed the PPMs and estimates class weights for
  the remaining lengths.
- Here unknown characters (e.g. 'X') are not supported, whereas in MixMHCp such
  characters affect the exponent in the prior term of the log likelihood.

References
    [1] David Gfeller, Philippe Guillaume, Justine Michaux, Hui-Song Pak, Roy
        T. Daniel, Julien Racle, George Coukos, Michal Bassani-Sternberg;
        The Length Distribution and Multiple Specificity of Naturally Presented
        HLA-I Ligands. J Immunol 15 December 2018; 201 (12): 3705-3716.
        https://doi.org/10.4049/jimmunol.1800914
"""
from __future__ import annotations

import numpy as np

from emmo.em.c_extensions.mhc1_c_ext import run_em
from emmo.em.mhc1_base import BaseEMRunnerMHC1
from emmo.pipeline.background import Background
from emmo.pipeline.sequences import SequenceManager


class EMRunnerMHC1(BaseEMRunnerMHC1):
    """Class for running the deconvolution of MHC1 ligands."""

    def __init__(
        self,
        sequence_manager: SequenceManager,
        motif_length: int,
        number_of_classes: int,
        n_term: int = 3,
        c_term: int = 2,
        n_term_penalty: float = 0.05,
        c_term_penalty: float = 0.2,
    ) -> None:
        """Initialize the class for running the deconvolution of MHC1 ligands.

        Args:
            sequence_manager: The instance holding the input sequences.
            motif_length: The length of the motif(s) to be estimated.
            number_of_classes: The number of motifs to be identified (not counting the flat motif).
            n_term: The length of the N-terminal part of the sequence to be aligned with the motif
                for sequences that are shorter/longer than 'motif_length'.
            c_term: The length of the C-terminal part of the sequence to be aligned with the motif
                for sequences that are shorter/longer than 'motif_length'.
            n_term_penalty: The penalty factor to be added for N-terminal extensions (sequence
                overhangs the alignment to motif at the N-terminus).
            c_term_penalty: The penalty factor to be added for C-terminal extensions (sequence
                overhangs the alignment to motif at the C-terminus).
        """
        super().__init__(
            sequence_manager,
            motif_length,
            number_of_classes,
            n_term=n_term,
            c_term=c_term,
            n_term_penalty=n_term_penalty,
            c_term_penalty=c_term_penalty,
        )

    def _expectation_maximization(self) -> None:
        """Do one expectation-maximization run.

        Does one EM run until convergence and sets the following attributes:
        - self.current_class_weights
        - self.current_ppm
        - self.current_score
        - self.current_responsibilities
        - self.current_steps

        Raises:
            RuntimeError: If the number of unique lengths does not match the number of row in the
                class weights array.
        """
        # first, the arrays to be passed to the C extension for the EM algorithm are prepared;
        # it expects the sequences to be sorted by length, so all corresponding arrays (sequences,
        # lengths, responsibilities) must be sorted by length

        ppm_array = np.zeros((self.n_classes, self.motif_length, self.n_alphabet))
        # the last class is the flat motif which remains unchanged
        ppm_array[self.number_of_classes] = Background("uniprot").frequencies

        seq_array = np.zeros((self.sm.number_of_sequences, self.sm.max_length), dtype=np.uint16)
        length_array = np.zeros(self.sm.number_of_sequences, dtype=np.uint16)

        for i, encoded_seq in enumerate(self.sm.sequences_as_indices()):
            seq_array[i, : len(encoded_seq)] = encoded_seq
            length_array[i] = len(encoded_seq)

        indices_sorted = np.argsort(length_array, kind="stable")
        indices_sorted_rev = np.argsort(indices_sorted)

        # sort the arrays by length
        seq_array = seq_array[indices_sorted]
        length_array = length_array[indices_sorted]
        responsibility_array = self._initialize_responsibilities()[indices_sorted]

        unique_lengths = np.unique(length_array)
        n_lengths = len(unique_lengths)

        # now run the EM algorithm in the C extension
        (
            ppm_array,
            responsibility_array,
            class_weight_array,
            _,  # n_positions
            _,  # c_positions
            self.current_score,
            self.current_steps,
        ) = run_em(
            ppm_array,
            responsibility_array,
            seq_array,
            length_array,
            self.motif_length,
            self.n_term,
            self.c_term,
            n_lengths,
            0,  # verbose is only used for debugging
            self.min_error,
            self.pseudocount,
            self.n_term_penalty,
            self.c_term_penalty,
        )

        # finally store the results in the instance attributes
        self.current_ppm = ppm_array

        if len(unique_lengths) != class_weight_array.shape[0]:
            raise RuntimeError(
                "the number of unique lengths does not match the number of row in the class "
                "weights array"
            )
        self.current_class_weights = {}
        for i, length in enumerate(unique_lengths):
            self.current_class_weights[length] = class_weight_array[i].copy()

        self.current_responsibilities = responsibility_array[indices_sorted_rev]
