"""Expectation-maximization-based deconvolution of MHC2 ligands.

This mostly follows the methods in Racle et al. 2019 with the associated tool MoDec. Modifications
are:
- A correction factor for the middle offset (position 0) to account for the fact that about half
  of the peptides contribute to this position (thoe for which the difference between their length
  and the motif length is even).

References:
    [1] Julien Racle, Justine Michaux, Georg Alexander Rockinger, Marion Arnaud, Sara Bobisse,
        Chloe Chong, Philippe Guillaume, George Coukos,  Alexandre Harari, Camilla Jandus, Michal
        Bassani-Sternberg, David Gfeller; Deep motif deconvolution of HLA-II peptidomes for robust
        class II epitope predictions. Nature Biotechnology 37, 1283–1286 (2019).
        https://doi.org/10.1038/s41587-019-0289-6
"""
from __future__ import annotations

import numpy as np

from emmo.em.mhc2_base import BaseEMRunnerMHC2
from emmo.em.mhc2_c_ext import run_em
from emmo.pipeline.background import BackgroundType
from emmo.pipeline.sequences import SequenceManager


class EMRunnerMHC2(BaseEMRunnerMHC2):
    """Class for running the deconvolution of MHC2 ligands."""

    def __init__(
        self,
        sequence_manager: SequenceManager,
        motif_length: int,
        number_of_classes: int,
        background: BackgroundType,
    ) -> None:
        """Initialize the class for running the deconvolution of MHC2 ligands.

        Args:
            sequence_manager: The instance holding the input sequences.
            motif_length: The length of the motif(s) to be estimated.
            number_of_classes: The number of motifs to be identified (not counting the flat motif).
            background: The background amino acid frequencies. Can also be a string corresponding
                to one of the available backgrounds.
        """
        super().__init__(
            sequence_manager,
            motif_length,
            number_of_classes,
            background,
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
        ppm_array[self.number_of_classes] = self.background.frequencies

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
        sim_weights_array = np.array(self.similarity_weights, dtype=np.float64)[indices_sorted]

        unique_lengths = np.sort(np.unique(length_array))

        offset_array = np.zeros(
            (len(unique_lengths), self.sm.max_length - self.motif_length + 1), dtype=np.uint16
        )
        for i, length in enumerate(unique_lengths):
            offsets = self.aligned_offsets.get_offset_list(length)
            offset_array[i, : len(offsets)] = offsets

        # now run the EM algorithm in the C extension
        (
            ppm_array,
            pssm_array,
            responsibility_array,
            class_weight_array,
            log_likelihood_ppm,
            log_likelihood_pssm,
            self.current_steps,
        ) = run_em(
            ppm_array,
            np.array(self.background.frequencies, dtype=np.float64),
            responsibility_array,
            seq_array,
            length_array,
            sim_weights_array,
            offset_array,
            self.motif_length,
            0,  # verbose is only used for debugging
            self.min_error,
            self.pseudocount,
            self.upweight_middle_offset,
        )

        # finally store the results in the instance attributes
        self.current_ppm = ppm_array
        self.current_pssm = pssm_array
        self.current_score = log_likelihood_pssm
        self.current_log_likelihood_ppm = log_likelihood_ppm

        if class_weight_array.shape != responsibility_array.shape[1:]:
            raise RuntimeError(
                "the shape of the class weights array does not match the number of classes and "
                "offsets"
            )
        self.current_class_weights = class_weight_array

        self.current_responsibilities = responsibility_array[indices_sorted_rev]
