"""Implementation of the EM algorithm for MHC2 ligands."""
from __future__ import annotations

from itertools import product

import numpy as np

from emmo.em.mhc2_base import BaseEMRunnerMHC2
from emmo.pipeline.sequences import SequenceManager


class EMRunnerMHC2(BaseEMRunnerMHC2):
    """Class for running the EM algorithm for MHC2 ligands."""

    def __init__(
        self,
        sequence_manager: SequenceManager,
        motif_length: int,
        number_of_classes: int,
        background: str = "MHC2_biondeep",
    ) -> None:
        """Initialize the MHC2 EM runner.

        Args:
            sequence_manager: The instance holding the input sequences.
            motif_length: The length of the motif(s) to be estimated.
            number_of_classes: The number of motifs to be identified (not counting the flat motif).
            background: The background amino acid frequencies. Must be a string corresponding to
                one of the available backgrounds.
        """
        super().__init__(
            sequence_manager,
            motif_length,
            number_of_classes,
            background=background,
        )

        self.offset_list_by_length = {
            length: self.aligned_offsets.get_offset_list(length)
            for length in self.similarity_weights_by_length
        }

    def _expectation_maximization(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, int,]:
        """Do one expectation-maximization run.

        Returns:
            The class weights, PPM, PSSM, log likelihood (using PSSM), log likelihood (using PPM),
            responsibilities, and EM steps.
        """
        # position probability matrices
        self.PPM = np.zeros((self.n_classes, self.motif_length, self.n_alphabet))
        # the last class is the flat motif which remains unchanged
        self.PPM[self.number_of_classes] = self.background_freqs

        # position-specific scoring matrix
        # in the expectation step and for the log likehood, we do not use the raw frequencies but
        # divide them by the background frequencies, we pre-compute these "scores" after the
        # maximization step to save time
        self.PSSM = self.PPM / self.background_freqs

        # probalitities of the classes and offsets
        self.class_weights = np.zeros((self.n_classes, self.n_offsets))

        self.responsibilities_by_length = self.sm.split_array_by_size(
            self._initialize_responsibilities()
        )

        # initialize PPMs and class weight based on initial responsibilities
        self._maximization()

        log_likelihood_pssm = self._log_likelihood(use_pssm=True)
        log_likelihood_error = float("inf")

        steps = 0

        while log_likelihood_error > self.min_error:
            steps += 1

            self._expectation()
            self._maximization()
            new_log_likelihood_pssm = self._log_likelihood(use_pssm=True)
            log_likelihood_error = abs(log_likelihood_pssm - new_log_likelihood_pssm)
            log_likelihood_pssm = new_log_likelihood_pssm

            print(
                f"Estimating frequencies, "
                f"{steps:4} EM steps, "
                f" score = {log_likelihood_pssm} ...",
                end="\r",
                flush=True,
            )

        log_likelihood_ppm = self._log_likelihood(use_pssm=False)

        return (
            self.class_weights,
            self.PPM,
            self.PSSM,
            log_likelihood_pssm,
            log_likelihood_ppm,
            self.sm.recombine_split_array(self.responsibilities_by_length),
            steps,
        )

    def _log_likelihood(self, use_pssm: bool = False) -> float:
        """Log likelihood given the current parameters.

        Like MoDec, we use a score computed from log odds ratios for the
        EM algorithm rather that loglikelihood.

        Args:
            use_pssm: Whether to use the PSSM (= PPM divided by background) instead of the PPM.

        Returns:
            Log likelihood.
        """
        matrix = self.PSSM if use_pssm else self.PPM
        log_likelihood = 0.0
        for length, sequences in self.sm.get_size_sorted_arrays().items():
            n_sequences = sequences.shape[0]
            offset_list = self.offset_list_by_length[length]
            similarity_weights = self.similarity_weights_by_length[length]

            p = np.zeros(n_sequences)

            for s, c, (i, o) in product(
                range(n_sequences), range(self.n_classes), enumerate(offset_list)
            ):
                prob = self.class_weights[c, o]
                for k in range(self.motif_length):
                    prob *= matrix[c, k, sequences[s, i + k]]
                p[s] += prob

            log_likelihood += np.sum(similarity_weights * np.log(p))

        # at the moment, we use pseudocount as a constant exponent of the Dirichlet priors, hence
        # we can do the following
        log_likelihood += self.pseudocount * np.sum(np.log(self.PPM[: self.number_of_classes]))

        return log_likelihood

    def _maximization(self) -> None:
        """Maximization step."""
        self.PPM[: self.number_of_classes] = self.pseudocount
        self.class_weights[:] = 0

        for length, sequences in self.sm.get_size_sorted_arrays().items():
            n_sequences = sequences.shape[0]
            offset_list = self.offset_list_by_length[length]
            similarity_weights = self.similarity_weights_by_length[length]
            responsibilities = self.responsibilities_by_length[length]

            for s, c, (i, o) in product(
                range(n_sequences), range(self.number_of_classes), enumerate(offset_list)
            ):
                resp = responsibilities[s, c, o] * similarity_weights[s]
                for k in range(self.motif_length):
                    self.PPM[c, k, sequences[s, i + k]] += resp

            self.class_weights[:] += np.sum(
                responsibilities * similarity_weights[:, np.newaxis, np.newaxis], axis=0
            )

        # normalize so that frequencies sum to one for each position
        self.PPM[: self.number_of_classes] /= np.sum(
            self.PPM[: self.number_of_classes], axis=2, keepdims=True
        )

        self.PSSM[: self.number_of_classes] = (
            self.PPM[: self.number_of_classes] / self.background_freqs
        )

        # upweight the middle offset
        self.class_weights[:, self.class_weights.shape[1] // 2] *= self.upweight_middle_offset

        self.class_weights[:] += self.pseudocount

        # normalize so that class weights sum to one
        self.class_weights /= np.sum(self.class_weights)

    def _expectation(self) -> None:
        """Expectation step."""
        for length, sequences in self.sm.get_size_sorted_arrays().items():
            n_sequences = sequences.shape[0]
            offset_list = self.offset_list_by_length[length]
            responsibilities = self.responsibilities_by_length[length]

            responsibilities[:] = 0

            for s, c, (i, o) in product(
                range(n_sequences), range(self.n_classes), enumerate(offset_list)
            ):
                prob = self.class_weights[c, o]
                for k in range(self.motif_length):
                    prob *= self.PSSM[c, k, sequences[s, i + k]]
                responsibilities[s, c, o] = prob

            responsibilities /= np.sum(responsibilities, axis=(1, 2), keepdims=True)


if __name__ == "__main__":
    from emmo.constants import REPO_DIRECTORY

    input_name = "HLA-A0101_A0218_background_class_II"
    directory = REPO_DIRECTORY / "validation" / "local"
    file = directory / f"{input_name}.txt"
    output_directory = directory / input_name

    sm = SequenceManager.load_from_txt(file)
    em_runner = EMRunnerMHC2(sm, 9, 2)
    em_runner.run(output_directory, output_all_runs=True, force=True)
