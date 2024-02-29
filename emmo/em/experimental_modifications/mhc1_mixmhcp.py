"""Implementation of the EM algorithm for MHC1 ligands (MixMHCp version).

This implementation follows more closely what MixMHCp does, i.e., only the ligands of the same
length as the motif are used to estimate the PPM. Then the class weights for the remaining lengths
are estimated.
"""
from __future__ import annotations

import itertools
from time import perf_counter

import numpy as np

from emmo.io.file import Openable
from emmo.io.output import write_matrices
from emmo.io.output import write_responsibilities
from emmo.pipeline.background import Background
from emmo.pipeline.sequences import SequenceManager


PSEUDO_COUNT_PRIOR = 0.1
MIN_ERROR = 0.001

N_TERM_PENALTY = 0.05
C_TERM_PENALTY = 0.2


class EqualLengthEM:
    """Class for EM using sequences of equal length as the motif."""

    def __init__(self, sequences: np.ndarray, len_alphabet: int, number_of_classes: int) -> None:
        """Initialize the EqualLengthEM class.

        Args:
            sequences: The encoded sequences.
            len_alphabet: The length of the alphabet.
            number_of_classes: The number of motifs to be estimated.
        """
        self.sequences = sequences

        # number of classes (excluding the flat motif)
        self.N = number_of_classes

        # we additionally want a flat motif
        self.n_classes = number_of_classes + 1

        self.n_alphabet = len_alphabet
        self.n_sequences = sequences.shape[0]
        self.n_motif = sequences.shape[1]

        self.b = Background("uniprot").frequencies

    def run(self, n_runs: int = 5, random_seed: int = 0) -> None:
        """Run the expectation-maximization algorithm.

        Args:
            n_runs: Number of EM runs.
            random_seed: The random seed to be used for initializing the EM runs.
        """
        # random number generator
        self.rng = np.random.default_rng(random_seed)

        self.best_run = None
        self.best_log_likelihood = float("-inf")

        for run in range(n_runs):
            start_time = perf_counter()

            self.pwm = np.zeros((self.n_classes, self.n_motif, self.n_alphabet))
            # the last class is the flat motif which remains unchanged
            self.pwm[self.N] = self.b

            self.class_weights = np.zeros(self.n_classes)
            self.responsibilities = np.zeros((self.n_sequences, self.n_classes))

            self.initialize_responsibilities()
            self.maximization()

            log_likelihood = self._log_likelihood()
            log_likelihood_error = float("inf")

            steps = 0

            while log_likelihood_error > MIN_ERROR:
                steps += 1

                self.expectation()
                self.maximization()
                new_log_likelihood = self._log_likelihood()
                log_likelihood_error = abs(log_likelihood - new_log_likelihood)
                log_likelihood = new_log_likelihood

                print(
                    f"Estimating frequencies (run {run:2}), "
                    f"{steps:4} EM steps, "
                    f" logL = {log_likelihood} ...",
                    end="\r",
                    flush=True,
                )

            print(
                f"Estimating frequencies (run {run:2}), "
                f"{steps:4} EM steps, "
                f" logL = {log_likelihood} ... finished {self.n_sequences} sequences "
                f"in {perf_counter()-start_time:.4f} seconds."
            )

            if log_likelihood > self.best_log_likelihood:
                self.best_run = run
                self.best_log_likelihood = log_likelihood
                self.best_pwm = self.pwm
                self.best_class_weights = self.class_weights
                self.best_responsibilities = self.responsibilities

        # finally sort classes by class weight in decreasing order but such that flat motif remains
        # the last class
        order = np.concatenate([np.flip(np.argsort(self.best_class_weights[: self.N])), [self.N]])
        self.best_class_weights = np.take(self.best_class_weights, order)
        self.best_pwm = np.take(self.best_pwm, order, axis=0)
        self.best_responsibilities = np.take(self.best_responsibilities, order, axis=1)

    def initialize_responsibilities(self) -> None:
        """Iniliatialize the responsibilities at random.

        Returns:
            The initialized responsibilities.
        """
        # initialize by assigning a random class to each sequence
        self.responsibilities[:] = 0

        random_classes = self.rng.integers(self.n_classes, size=self.n_sequences, dtype=np.uint32)

        # ensure that at least one sequence is assigned to each classes
        for c, s in enumerate(
            self.rng.choice(
                self.n_sequences,
                size=min(self.n_sequences, self.n_classes),
                replace=False,
                shuffle=True,
            )
        ):
            random_classes[s] = c

        for i in range(self.n_sequences):
            self.responsibilities[i, random_classes[i]] = 1.0

    def _log_likelihood(self) -> float:
        """Compute the log likelihood under the current model.

        Returns:
            Log likelihood.
        """
        log_likelihood = 0.0

        p = np.zeros(self.n_sequences)
        for s in range(self.n_sequences):
            for c in range(self.n_classes):
                sp = 1.0
                for pos, a in enumerate(self.sequences[s]):
                    sp *= self.pwm[c, pos, a]
                p[s] += self.class_weights[c] * sp

        log_likelihood += np.sum(np.log(p))

        # at the moment, we use PSEUDO_COUNT_PRIOR as a constant exponent of the Dirichlet priors,
        # hence we can do the following
        log_likelihood += PSEUDO_COUNT_PRIOR * np.sum(np.log(self.pwm[: self.N]))

        return log_likelihood

    def maximization(self) -> None:
        """Maximization step."""
        # maximization step for PWM
        self.pwm[: self.N] = PSEUDO_COUNT_PRIOR

        for c in range(self.N):
            for s, resp in enumerate(self.responsibilities[:, c]):
                for pos, a in enumerate(self.sequences[s]):
                    self.pwm[c, pos, a] += resp

            # normalize so that frequencies sum to one for each position
            self.pwm[c] /= np.sum(self.pwm[c], axis=1)[:, np.newaxis]

        # maximization step for class weights
        self.class_weights[:] = np.sum(self.responsibilities, axis=0)
        self.class_weights /= sum(self.class_weights)

    def expectation(self) -> None:
        """Expectation step."""
        self.responsibilities[:] = self.class_weights

        for s in range(self.n_sequences):
            for c in range(self.n_classes):
                for pos, a in enumerate(self.sequences[s]):
                    self.responsibilities[s, c] *= self.pwm[c, pos, a]

        self.responsibilities /= np.sum(self.responsibilities, axis=1)[:, np.newaxis]

    def get_responsibilities(self) -> np.ndarray:
        """The best responsibilities.

        Returns:
            The best responsibilities.
        """
        return self.best_responsibilities

    def get_number_of_parameters(self) -> int:
        """Return the number of parameters (frequencies and class priors).

        Returns:
            The number of parameters.
        """
        # class weight incl. flat motif (subtract 1 because weights sum to one)
        count = self.n_classes - 1

        # at the moment the flat motif is fix and not estimated
        # count += self.n_alphabet

        # frequencies summed over all other classes (subtract 1 because aa frequencies to one)
        count += self.N * self.n_motif * (self.n_alphabet - 1)

        return count

    def get_log_likelihood(self) -> float:
        """Return the best log likelihood.

        Returns:
            Log likelihood.
        """
        return self.best_log_likelihood

    def get_aic(self) -> float:
        """Return the Akaike information criterion (AIC) using the log likelihood of the best run.

        Returns:
            AIC.
        """
        return 2 * self.get_number_of_parameters() - 2 * self.get_log_likelihood()


class ClassWeightsEM:
    """Class for estimating the class weights for lengths other than the motif length."""

    def __init__(self, sequences: np.ndarray, pwm: np.ndarray, n_term: int, c_term: int) -> None:
        """Initialize the ClassWeightsEM class.

        Args:
            sequences: The encoded sequences.
            pwm: The position weight matrix.
            n_term: The length of the N-terminal part of the sequence to be aligned with the motif
                for sequences that are shorter/longer than 'motif_length'.
            c_term: The length of the C-terminal part of the sequence to be aligned with the motif
                for sequences that are shorter/longer than 'motif_length'.
        """
        self.sequences = sequences
        self.pwm = pwm
        self.n_term = n_term
        self.c_term = c_term

        self.n_sequences, self.len_seq = self.sequences.shape
        self.n_classes, self.n_motif, self.n_alphabet = pwm.shape

        # N is the number of classes excluding the flat motif
        self.N = self.n_classes - 1

    def run(self) -> None:
        """Run the EM algorithm."""
        start_time = perf_counter()

        self.n_positions = np.zeros((self.n_sequences, self.n_classes), dtype=np.int8)
        self.c_positions = np.full_like(self.n_positions, self.len_seq - self.c_term)

        self.class_weights = np.full(self.n_classes, 1 / self.n_classes)
        self.responsibilities = np.zeros((self.n_sequences, self.n_classes))

        self.log_likelihood = float("-inf")
        log_likelihood_error = float("inf")
        steps = 0

        while log_likelihood_error > MIN_ERROR:
            steps += 1
            self.expectation()
            self.maximization()
            new_log_likelihood = self._log_likelihood()
            log_likelihood_error = abs(self.log_likelihood - new_log_likelihood)
            self.log_likelihood = new_log_likelihood

            print(
                f"Estimating weights (length {self.len_seq:3}), "
                f"{steps:4} EM steps, "
                f" logL = {self.log_likelihood} ...",
                end="\r",
                flush=True,
            )

        print(
            f"Estimating weights (length {self.len_seq:3}), "
            f"{steps:4} EM steps, "
            f" logL = {self.log_likelihood} ... finished {self.n_sequences} sequences "
            f"in {perf_counter()-start_time:.4f} seconds."
        )

    def _log_likelihood(self) -> float:
        """Compute the log likelihood under the current model.

        Returns:
            Log likelihood.
        """
        p = np.zeros(self.n_sequences)
        for s in range(self.n_sequences):
            for c in range(self.n_classes):
                sp = self.class_weights[c]
                n_start = self.n_positions[s, c]
                c_start = self.c_positions[s, c]

                # contribution of the N-terminal part of the motif
                for pos in range(n_start, n_start + self.n_term):
                    sp *= self.pwm[c, pos - n_start, self.sequences[s, pos]]

                # contribution of the C-terminal part of the motif
                for pos in range(c_start, c_start + self.c_term):
                    # position in the motif = motif length -
                    # (C-term. start + length C-term. - position)
                    sp *= self.pwm[
                        c, self.n_motif - c_start - self.c_term + pos, self.sequences[s, pos]
                    ]

                p[s] += sp

        # no log priors added since PWM is fixed here

        return np.sum(np.log(p))

    def maximization(self) -> None:
        """Maximization step."""
        self.class_weights[:] = np.sum(self.responsibilities, axis=0)
        self.class_weights /= sum(self.class_weights)

    def expectation(self) -> None:
        """Expectation step."""
        if self.len_seq < self.n_motif:
            self._expectation_shorter_than_motif()
        else:
            self._expectation_longer_than_motif()

        self.responsibilities /= np.sum(self.responsibilities, axis=1)[:, np.newaxis]

    def _expectation_shorter_than_motif(self) -> None:
        """Expectation step for sequences shorter than the motif."""
        self.responsibilities[:] = self.class_weights

        for s, c in itertools.product(range(self.n_sequences), range(self.n_classes)):
            for pos in range(self.n_term):
                self.responsibilities[s, c] *= self.pwm[c, pos, self.sequences[s, pos]]

            for pos in range(self.len_seq - self.c_term, self.len_seq):
                # position in the motif = motif length - (seq. length - position)
                self.responsibilities[s, c] *= self.pwm[
                    c, self.n_motif - self.len_seq + pos, self.sequences[s, pos]
                ]

    def _expectation_longer_than_motif(self) -> None:  # noqa: CCR001
        """Expectation step for sequences longer than the motif."""
        for s, c in itertools.product(range(self.n_sequences), range(self.n_classes)):
            current_max = 0

            for n_start in range(self.len_seq - self.n_motif + 1):
                # penalty if motif does not start at first position
                resp_n_term = N_TERM_PENALTY**n_start

                for pos in range(self.n_term):
                    resp_n_term *= self.pwm[c, pos, self.sequences[s, pos + n_start]]

                for c_start in range(
                    n_start + self.n_motif - self.c_term, self.len_seq - self.c_term + 1
                ):
                    # penalty if motif does not end at last position
                    resp_c_term = C_TERM_PENALTY ** (self.len_seq - c_start - self.c_term)
                    for pos in range(self.c_term):
                        resp_c_term *= self.pwm[
                            c,
                            self.n_motif - self.c_term + pos,
                            self.sequences[s, pos + c_start],
                        ]

                    if resp_n_term * resp_c_term > current_max:
                        current_max = resp_n_term * resp_c_term
                        self.n_positions[s, c] = n_start
                        self.c_positions[s, c] = c_start

            self.responsibilities[s, c] = self.class_weights[c] * current_max

    def get_responsibilities(self) -> np.ndarray:
        """The responsibilities.

        Returns:
            The responsibilities.
        """
        return self.responsibilities

    def get_log_likelihood(self) -> float:
        """Return the log likelihood.

        Returns:
            Log likelihood.
        """
        return self.log_likelihood

    def get_number_of_parameters(self) -> int:
        """Return the number of parameters (class priors).

        Returns:
            The number of parameters.
        """
        # class weight incl. flat motif (subtract 1 because weights sum to one)
        return self.n_classes - 1

    def get_aic(self) -> float:
        """Return the Akaike information criterion (AIC).

        Returns:
            AIC.
        """
        return 2 * self.get_number_of_parameters() - 2 * self.get_log_likelihood()


class FullEM:
    """Class for the full method combining the results of all lengths."""

    def __init__(
        self, sequence_manager: SequenceManager, motif_length: int, number_of_classes: int
    ) -> None:
        """Initialize the FullEM class.

        Args:
            sequence_manager: The instance holding the input sequences.
            motif_length: The length of the motif(s) to be estimated.
            number_of_classes: The number of motifs to be identified (not counting the flat motif).

        Raises:
            ValueError: If there are no sequences of equal length as the motif.
        """
        self.sm = sequence_manager
        self.motif_length = motif_length
        self.N = number_of_classes

        if self.motif_length not in self.sm.get_size_sorted_sequences():
            raise ValueError(
                f"no sequence with the specified motif length " f"{self.motif_length} found"
            )

    def run(self, random_seed: int = 0) -> None:
        """Run the expectation-maximization algorithm.

        Args:
            random_seed: The random seed to be used for initializing the EM runs.
        """
        start_time = perf_counter()

        print(
            "Starting PWM estimation (using the sequences of the same" " length as the motif) ..."
        )

        self.main_EM_runner = EqualLengthEM(
            self.sm.get_size_sorted_arrays()[self.motif_length], len(self.sm.alphabet), self.N
        )

        self.main_EM_runner.run(random_seed=random_seed)
        self.pwm = self.main_EM_runner.best_pwm

        print(f"Finished frequency estimation in {perf_counter() - start_time}" " seconds.")

        print("Starting class weight estimation for other lengths ...")
        self.runners: dict[int, EqualLengthEM | ClassWeightsEM] = {
            self.motif_length: self.main_EM_runner
        }

        for length, seq_array in self.sm.get_size_sorted_arrays().items():
            if length == self.motif_length:
                continue

            runner = ClassWeightsEM(seq_array, self.pwm, 3, 2)
            runner.run()
            self.runners[length] = runner

        print(f"Finished everything in {perf_counter() - start_time} seconds.")

    def get_log_likelihood(self) -> float:
        """Return the log likelihood.

        Returns:
            Log likelihood.
        """
        return sum(r.get_log_likelihood() for r in self.runners.values())

    def get_number_of_parameters(self) -> int:
        """Return the number of parameters (PWMs and class priors for all lengths).

        Returns:
            The number of parameters.
        """
        return sum(r.get_number_of_parameters() for r in self.runners.values())

    def write_results(self, directory: Openable, force: bool = True) -> None:
        """Write the results into a directory.

        Args:
            directory: The output directory.
            force: Overwrite files if they already exist.
        """
        write_matrices(
            directory, self.pwm[:-1], self.sm.alphabet, flat_motif=self.pwm[-1][0], force=force
        )

        write_responsibilities(
            directory,
            self.sm,
            {length: runner.get_responsibilities() for length, runner in self.runners.items()},
            self.N,
            force=force,
        )


if __name__ == "__main__":
    from emmo.constants import REPO_DIRECTORY

    # input_name = 'HLA-A0101_A0218_background'
    input_name = "HLA-A0101_A0218_background_various_lengths"
    directory = REPO_DIRECTORY / "validation" / "local"
    file = directory / f"{input_name}.txt"
    output_directory = directory / input_name

    sm = SequenceManager.load_from_txt(file)
    em_runner = FullEM(sm, 9, 2)
    em_runner.run()

    em_runner.write_results(output_directory, force=True)
