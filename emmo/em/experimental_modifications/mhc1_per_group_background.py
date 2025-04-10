"""Expectation-maximization-based deconvolution of MHC1 ligands with group-specific background.

This module implements a modified version of the MHC1 deconvolution algorithm that uses group-
and position-specific background amino acid distributions in the log likelihood function. The
background distribution are provided as position probability matrices (PPMs) for each group.
"""
from __future__ import annotations

import itertools

import numpy as np

from emmo.constants import MHC1_ALLELE_COL
from emmo.em.mhc1 import EMRunnerMHC1
from emmo.models.deconvolution import DeconvolutionModelMHC1
from emmo.pipeline.sequences import SequenceManager


class EMRunnerMHC1PerGroupBackground(EMRunnerMHC1):
    """Class for running the deconvolution of MHC1 ligands with group-specific background."""

    def __init__(
        self,
        sequence_manager: SequenceManager,
        motif_length: int,
        number_of_classes: int,
        group2ppm: dict[str, np.ndarray],
        n_term: int = 3,
        c_term: int = 2,
        n_term_penalty: float = 0.05,
        c_term_penalty: float = 0.2,
        group_attribute: str = MHC1_ALLELE_COL,
    ) -> None:
        """Initialize the class for running the deconvolution.

        Args:
            sequence_manager: The instance holding the input sequences.
            motif_length: The length of the motif(s) to be estimated.
            number_of_classes: The number of motifs to be identified (not counting the flat motif).
            group2ppm: Dictionary mapping group names to position probability matrices (PPMs) that
                will be used as group-specific background.
            n_term: The length of the N-terminal part of the sequence to be aligned with the motif
                for sequences that are shorter/longer than 'motif_length'.
            c_term: The length of the C-terminal part of the sequence to be aligned with the motif
                for sequences that are shorter/longer than 'motif_length'.
            n_term_penalty: The penalty factor to be added for N-terminal extensions (sequence
                overhangs the alignment to motif at the N-terminus).
            c_term_penalty: The penalty factor to be added for C-terminal extensions (sequence
                overhangs the alignment to motif at the C-terminus).
            group_attribute: The name of the attribute in sequence_manager that holds the group
                information.
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

        if set(self.sm.size_sorted_sequences.keys()) != {motif_length}:
            raise ValueError(
                f"all sequences must have the same length as the motif (set to {motif_length})"
            )

        if not hasattr(sequence_manager, group_attribute):
            raise ValueError(f"SequenceManager has no attribute '{group_attribute}'.")
        self.groups = getattr(sequence_manager, group_attribute)
        self.groups_by_length = self.sm.get_size_sorted_features(group_attribute)
        self.group2ppm = group2ppm

        missing_groups = set(self.groups) - set(self.group2ppm.keys())
        if missing_groups:
            raise ValueError(f"Missing PPMs for groups: {missing_groups}")

        self.current_background_ppm: np.ndarray | None = None

    def _expectation_maximization(self) -> None:
        """Override the function for doing one expectation-maximization run.

        The function additionally initializes the background PPMs for the classes.
        """
        self.current_background_ppm = np.zeros((self.n_classes, self.motif_length, self.n_alphabet))
        super()._expectation_maximization()

    def _loglikelihood(self) -> float:  # noqa: CCR001
        """Override the function for computing the log likelihood under the current model.

        Returns:
            Log likelihood.
        """
        log_likelihood = 0.0
        for length, sequences in self.sm.size_sorted_arrays.items():
            n_sequences = sequences.shape[0]
            class_weights = self.class_weights_by_length[length]
            groups = self.groups_by_length[length]

            p = np.zeros(n_sequences)

            for s, c in itertools.product(range(n_sequences), range(self.n_classes)):
                background_ppm = self.group2ppm[groups[s]]

                # if the sequence and motif length are equal, we want the whole sequence to
                # contribute
                if length == self.motif_length:
                    prob = class_weights[c]
                    for pos, a in enumerate(sequences[s]):
                        prob *= self.current_ppm[c, pos, a] / background_ppm[pos, a]
                    p[s] += prob
                    continue

                # otherwise, only the N- and C-terminal positions contribute
                raise NotImplementedError(
                    "sequences shorter/longer than the motif are not supported yet"
                )

            log_likelihood += np.sum(np.log(p))

        # at the moment, we use pseudocount as a constant exponent of the Dirichlet priors, hence
        # we can do the following
        log_likelihood += self.pseudocount * np.sum(
            np.log(self.current_ppm[: self.number_of_classes])
        )

        return log_likelihood

    def _maximization(self) -> None:  # noqa: CCR001
        """Override the function for the maximization step."""
        self.current_ppm[: self.number_of_classes] = self.pseudocount
        self.current_background_ppm[:] = 0.0

        for length, sequences in self.sm.size_sorted_arrays.items():
            n_sequences = sequences.shape[0]
            class_weights = self.class_weights_by_length[length]
            responsibilities = self.responsibilities_by_length[length]
            groups = self.groups_by_length[length]

            for s, c in itertools.product(range(n_sequences), range(self.n_classes)):
                resp = responsibilities[s, c]
                background_ppm = self.group2ppm[groups[s]]

                self.current_background_ppm[c] += resp * background_ppm

                if c >= self.number_of_classes:
                    continue

                # if the sequence and motif length are equal, we want the whole sequence to
                # contribute
                if length == self.motif_length:
                    for pos, a in enumerate(sequences[s]):
                        self.current_ppm[c, pos, a] += resp
                    continue

                # otherwise, only the N- and C-terminal positions contribute
                raise NotImplementedError(
                    "sequences shorter/longer than the motif are not supported yet"
                )

            # maximization step for class weights
            class_weights[:] = np.sum(responsibilities, axis=0)
            class_weights /= np.sum(class_weights)

        # normalize so that frequencies sum to one for each position
        for c in range(self.number_of_classes):
            self.current_ppm[c] /= np.sum(self.current_ppm[c], axis=1)[:, np.newaxis]
        for c in range(self.n_classes):
            self.current_background_ppm[c] /= np.sum(self.current_background_ppm[c], axis=1)[
                :, np.newaxis
            ]

    def _expectation(self) -> None:
        """Override the function for the expectation step."""
        for length in self.sm.size_sorted_arrays.keys():
            if length == self.motif_length:
                self._expectation_equal_to_motif()
            elif length < self.motif_length:
                raise NotImplementedError("sequences shorter than the motif are not supported yet")
            else:
                raise NotImplementedError("sequences longer than the motif are not supported yet")

    def _expectation_equal_to_motif(self) -> None:
        """Override the function for the expectation step (same length as the motif)."""
        length = self.motif_length
        sequences = self.sm.size_sorted_arrays[length]
        n_sequences = sequences.shape[0]
        class_weights = self.class_weights_by_length[length]
        responsibilities = self.responsibilities_by_length[length]
        groups = self.groups_by_length[length]

        for s, c in itertools.product(range(n_sequences), range(self.n_classes)):
            background_ppm = self.group2ppm[groups[s]]

            prob = class_weights[c]
            for pos, a in enumerate(sequences[s]):
                prob *= self.current_ppm[c, pos, a] / background_ppm[pos, a]
            responsibilities[s, c] = prob

        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

    def _build_model(self) -> DeconvolutionModelMHC1:
        """Override the method to collect the current parameters.

        The built model additionally contains the computed background PPMs for each class. These
        are weighted averages of the group-specific PPMs based on the responsibilities.

        Returns:
            The current parameters (PPMs, class weights, etc.) collected in an instance of the
            corresponding deconvolution model class.
        """
        model = super()._build_model()
        model.artifacts["background_ppm"] = self.current_background_ppm

        return model
