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

import itertools

import numpy as np

from emmo.em.mhc1_base import BaseEMRunnerMHC1
from emmo.io.sequences import SequenceManager
from emmo.resources.background_freqs import get_background


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

    def _expectation_maximization(
        self,
    ) -> tuple[dict[int, np.ndarray], np.ndarray, float, np.ndarray, int,]:
        """Do one expectation-maximization run.

        Returns:
            The class weights, PPM, log likelihood, responsibilities, and EM
            steps.
        """
        self.ppm = np.zeros((self.n_classes, self.motif_length, self.n_alphabet))
        # the last class is the flat motif which remains unchanged
        self.ppm[self.number_of_classes] = get_background(which="uniprot")

        self.responsibilities_by_length = self.sm.split_array_by_size(
            self._initialize_responsibilities()
        )

        self.class_weights_by_length = {}
        self.n_positions_by_length = {}
        self.c_positions_by_length = {}
        for length, sequences in self.sm.get_size_sorted_sequences().items():
            self.class_weights_by_length[length] = np.zeros(self.n_classes)
            self.n_positions_by_length[length] = np.zeros(
                (len(sequences), self.n_classes), dtype=np.int8
            )
            self.c_positions_by_length[length] = np.full_like(
                self.n_positions_by_length[length], length - self.c_term
            )

        self._maximization()

        log_likelihood = self._loglikelihood()
        log_likelihood_error = float("inf")
        steps = 0

        while log_likelihood_error > self.min_error:
            steps += 1

            self._expectation()
            self._maximization()
            new_log_likelihood = self._loglikelihood()
            log_likelihood_error = abs(log_likelihood - new_log_likelihood)
            log_likelihood = new_log_likelihood

            print(
                f"Estimating frequencies, {steps:4} EM steps, logL = {log_likelihood} ...",
                end="\r",
                flush=True,
            )

        return (
            self.class_weights_by_length,
            self.ppm,
            log_likelihood,
            self.sm.recombine_split_array(self.responsibilities_by_length),
            steps,
        )

    def _loglikelihood(self) -> float:  # noqa: CCR001
        """Compute the log likelihood under the current model.

        Returns:
            Log likelihood.
        """
        log_likelihood = 0.0
        for length, sequences in self.sm.get_size_sorted_arrays().items():
            n_sequences = sequences.shape[0]
            class_weights = self.class_weights_by_length[length]
            n_pos = self.n_positions_by_length[length]
            c_pos = self.c_positions_by_length[length]

            p = np.zeros(n_sequences)

            for s, c in itertools.product(range(n_sequences), range(self.n_classes)):
                # if the sequence and motif length are equal, we want the whole sequence to
                # contribute
                if length == self.motif_length:
                    prob = class_weights[c]
                    for pos, a in enumerate(sequences[s]):
                        prob *= self.ppm[c, pos, a]
                    p[s] += prob
                    continue

                # otherwise, only the N- and C-terminal positions contribute
                n_start = n_pos[s, c]
                c_start = c_pos[s, c]
                prob = (
                    class_weights[c]
                    * (self.n_term_penalty**n_start)
                    * (self.c_term_penalty ** (length - c_start - self.c_term))
                )

                # contribution of the N-terminal part of the motif
                for pos in range(n_start, n_start + self.n_term):
                    prob *= self.ppm[c, pos - n_start, sequences[s, pos]]

                # contribution of the C-terminal part of the motif
                for pos in range(c_start, c_start + self.c_term):
                    # position in the motif = motif length -
                    # (C-term. start + length C-term. - position)
                    prob *= self.ppm[
                        c, self.motif_length - c_start - self.c_term + pos, sequences[s, pos]
                    ]

                p[s] += prob

            log_likelihood += np.sum(np.log(p))

        # at the moment, we use pseudocount as a constant exponent of the Dirichlet priors, hence
        # we can do the following
        log_likelihood += self.pseudocount * np.sum(np.log(self.ppm[: self.number_of_classes]))

        return log_likelihood

    def _maximization(self) -> None:  # noqa: CCR001
        """Maximization step."""
        self.ppm[: self.number_of_classes] = self.pseudocount

        for length, sequences in self.sm.get_size_sorted_arrays().items():
            n_sequences = sequences.shape[0]
            class_weights = self.class_weights_by_length[length]
            n_pos = self.n_positions_by_length[length]
            c_pos = self.c_positions_by_length[length]
            responsibilities = self.responsibilities_by_length[length]

            for s, c in itertools.product(range(n_sequences), range(self.number_of_classes)):
                # if the sequence and motif length are equal, we want the whole sequence to
                # contribute
                if length == self.motif_length:
                    resp = responsibilities[s, c]
                    for pos, a in enumerate(sequences[s]):
                        self.ppm[c, pos, a] += resp
                    continue

                # otherwise, only the N- and C-terminal positions contribute
                n_start = n_pos[s, c]
                c_start = c_pos[s, c]
                resp = responsibilities[s, c]

                # contribution of the N-terminal part of the motif
                for pos in range(n_start, n_start + self.n_term):
                    self.ppm[c, pos - n_start, sequences[s, pos]] += resp

                # contribution of the C-terminal part of the motif
                for pos in range(c_start, c_start + self.c_term):
                    # position in the motif = motif length -
                    # (C-term. start + length C-term. - position)
                    self.ppm[
                        c, self.motif_length - c_start - self.c_term + pos, sequences[s, pos]
                    ] += resp

            # maximization step for class weights
            class_weights[:] = np.sum(responsibilities, axis=0)
            class_weights /= np.sum(class_weights)

        # normalize so that frequencies sum to one for each position
        for c in range(self.number_of_classes):
            self.ppm[c] /= np.sum(self.ppm[c], axis=1)[:, np.newaxis]

    def _expectation(self) -> None:
        """Expectation step."""
        for length in self.sm.get_size_sorted_arrays().keys():
            if length == self.motif_length:
                self._expectation_equal_to_motif()
            elif length < self.motif_length:
                self._expectation_shorter_than_motif(length)
            else:
                self._expectation_longer_than_motif(length)

    def _expectation_equal_to_motif(self) -> None:
        """Expectation step for sequences of the same length as the motif."""
        length = self.motif_length
        sequences = self.sm.get_size_sorted_arrays()[length]
        n_sequences = sequences.shape[0]
        class_weights = self.class_weights_by_length[length]
        responsibilities = self.responsibilities_by_length[length]

        for s, c in itertools.product(range(n_sequences), range(self.n_classes)):
            x = class_weights[c]
            for pos, a in enumerate(sequences[s]):
                x *= self.ppm[c, pos, a]
            responsibilities[s, c] = x

        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

    def _expectation_shorter_than_motif(self, length: int) -> None:
        """Expectation step for sequences shorter than the motif.

        Args:
            length: Length of the sequences.
        """
        sequences = self.sm.get_size_sorted_arrays()[length]
        n_sequences = sequences.shape[0]
        class_weights = self.class_weights_by_length[length]
        responsibilities = self.responsibilities_by_length[length]

        for s, c in itertools.product(range(n_sequences), range(self.n_classes)):
            x = class_weights[c]

            for pos in range(self.n_term):
                x *= self.ppm[c, pos, sequences[s, pos]]

            for pos in range(length - self.c_term, length):
                # position in the motif = motif length - (seq. length - position)
                x *= self.ppm[c, self.motif_length - length + pos, sequences[s, pos]]

            responsibilities[s, c] = x

        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)

    def _expectation_longer_than_motif(self, length: int) -> None:  # noqa: CCR001
        """Expectation step for sequences longer than the motif.

        Args:
            length: Length of the sequences.
        """
        sequences = self.sm.get_size_sorted_arrays()[length]
        n_sequences = sequences.shape[0]
        class_weights = self.class_weights_by_length[length]
        responsibilities = self.responsibilities_by_length[length]
        n_pos = self.n_positions_by_length[length]
        c_pos = self.c_positions_by_length[length]

        for s, c in itertools.product(range(n_sequences), range(self.n_classes)):
            current_max = 0

            for n_start in range(length - self.motif_length + 1):
                for c_start in range(
                    n_start + self.motif_length - self.c_term, length - self.c_term + 1
                ):
                    prob = (
                        class_weights[c]
                        * (self.n_term_penalty**n_start)
                        * (self.c_term_penalty ** (length - c_start - self.c_term))
                    )

                    for pos in range(self.n_term):
                        prob *= self.ppm[c, pos, sequences[s, pos + n_start]]

                    for pos in range(self.c_term):
                        prob *= self.ppm[
                            c,
                            self.motif_length - self.c_term + pos,
                            sequences[s, pos + c_start],
                        ]

                    if prob > current_max:
                        current_max = prob
                        n_pos[s, c] = n_start
                        c_pos[s, c] = c_start

            responsibilities[s, c] = current_max

        responsibilities /= np.sum(responsibilities, axis=1, keepdims=True)


if __name__ == "__main__":
    from emmo.models.deconvolution import DeconvolutionModelMHC1
    from emmo.constants import REPO_DIRECTORY

    input_name = "HLA-A0101_A0218_background_various_lengths"
    directory = REPO_DIRECTORY / "validation" / "local"
    file = directory / f"{input_name}.txt"
    output_directory = directory / input_name
    output_directory.mkdir(parents=True, exist_ok=True)

    sm = SequenceManager(file)
    em_runner = EMRunnerMHC1(sm, 9, 2)
    em_runner.run(output_directory, output_all_runs=True)

    model = em_runner.best_model
    for length, weights in model.class_weights.items():
        print(length, weights)

    reloaded_model = DeconvolutionModelMHC1.load(output_directory)
    for length, weights in reloaded_model.class_weights.items():
        print(length, weights)

    print(np.allclose(model.ppm, reloaded_model.ppm))
