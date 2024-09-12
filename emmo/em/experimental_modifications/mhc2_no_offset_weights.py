"""Implementation of the EM algorithm for MHC2 ligands without offset weights."""
from __future__ import annotations

from itertools import product
from time import perf_counter

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from emmo.io.file import Openable
from emmo.io.file import save_csv
from emmo.models.deconvolution import DeconvolutionModelMHC2NoOffsetWeights
from emmo.pipeline.background import Background
from emmo.pipeline.background import BackgroundType
from emmo.pipeline.sequences import SequenceManager
from emmo.utils.statistics import compute_aic


class _EqualLengthHandler:
    """Class for log likelihood, expectation, and maximization for sequences of equal length."""

    def __init__(
        self,
        model: DeconvolutionModelMHC2NoOffsetWeights,
        sequences: np.ndarray,
        similarity_weight: np.ndarray,
    ) -> None:
        """Initialize the class _EqualLengthHandler.

        Args:
            model: The deconvolution model instance.
            sequences: The encoded sequences.
            similarity_weight: The sequence weights.
        """
        self.model = model
        self.sequences = sequences
        self.n_sequences, self.len_seq = self.sequences.shape

        # arrays for position-specific scoring matrix and class weights are shared among the
        # instances of this class
        self.ppm = model.ppm
        self.pssm = model.pssm
        self.class_weights = model.class_weights

        self.n_classes, self.motif_length, self.n_alphabet = self.pssm.shape
        self.number_of_classes = self.n_classes - 1

        self.similarity_weight = similarity_weight

        self.n_offsets = self.len_seq - self.motif_length + 1

    def _initialize(self, rng: np.random._generator.Generator) -> None:
        """Initialize the responsibilities at random.

        Returns:
            The initialized responsibilities.
        """
        # initialize responsibilities by assigning a random class to each seq.
        self.responsibilities = np.zeros((self.n_sequences, self.n_classes, self.n_offsets))

        random_classes = rng.integers(self.n_classes, size=self.n_sequences, dtype=np.uint32)

        # ensure that at least one sequence is assigned to each classes
        for c, s in enumerate(
            rng.choice(
                self.n_sequences,
                size=min(self.n_sequences, self.n_classes),
                replace=False,
                shuffle=True,
            )
        ):
            random_classes[s] = c

        # uniform responsibility for all possible offsets
        for s in range(self.n_sequences):
            self.responsibilities[s, random_classes[s], :] = 1

        # normalize the initial responsibilities
        self.responsibilities /= np.sum(self.responsibilities, axis=(1, 2), keepdims=True)

    def _initialize_with_known_classes(self, classes: list, class_mapping: dict[str, int]) -> None:
        """Initialize the responsibilities with known classes.

        Returns:
            The initialized responsibilities.
        """
        self.responsibilities = np.zeros((self.n_sequences, self.n_classes, self.n_offsets))

        for s in range(self.n_sequences):
            self.responsibilities[s, class_mapping[classes[s]], :] = 1

        # normalize the initial responsibilities
        self.responsibilities /= np.sum(self.responsibilities, axis=(1, 2), keepdims=True)

    def _log_score(self) -> float:
        """Compute the log score using the current model.

        Like MoDec, we use a score computed from log odds ratios for the
        EM algorithm rather that loglikelihood.

        Returns:
            Score.
        """
        p = np.zeros(self.n_sequences)

        for s, c, i in product(
            range(self.n_sequences), range(self.n_classes), range(self.n_offsets)
        ):
            prob = self.class_weights[c]
            for k in range(self.motif_length):
                prob *= self.pssm[c, k, self.sequences[s, i + k]]
            p[s] += prob

        return np.sum(self.similarity_weight * np.log(p))

    def _log_likelihood(self) -> float:
        """Log likelihood given the current parameters.

        Returns:
            Log likelihood.
        """
        p = np.zeros(self.n_sequences)

        for s, c, i in product(
            range(self.n_sequences), range(self.n_classes), range(self.n_offsets)
        ):
            prob = self.class_weights[c]
            for k in range(self.motif_length):
                prob *= self.ppm[c, k, self.sequences[s, i + k]]
            p[s] += prob

        return np.sum(self.similarity_weight * np.log(p))

    def _maximization(self, ppm: np.ndarray) -> None:
        """Maximization step."""
        # pseudo counts have been added in the class 'VariableLengthEM'

        for s, c, i in product(
            range(self.n_sequences), range(self.number_of_classes), range(self.n_offsets)
        ):
            resp = self.responsibilities[s, c, i] * self.similarity_weight[s]
            for k in range(self.motif_length):
                ppm[c, k, self.sequences[s, i + k]] += resp

        self.class_weights[:] += np.sum(
            np.sum(self.responsibilities, axis=2) * self.similarity_weight[:, np.newaxis], axis=0
        )

        # normalization of PPM  and class weights happens when
        # responsibilities have been added for all different lengths

    def _expectation(self) -> None:
        """Expectation step."""
        self.responsibilities[:] = 0

        for s, c, i in product(
            range(self.n_sequences), range(self.n_classes), range(self.n_offsets)
        ):
            prob = self.class_weights[c]
            for k in range(self.motif_length):
                prob *= self.pssm[c, k, self.sequences[s, i + k]]
            self.responsibilities[s, c, i] = prob

        self.responsibilities /= np.sum(self.responsibilities, axis=(1, 2), keepdims=True)


class VariableLengthEM:
    """Class for processing all different length."""

    def __init__(
        self,
        sequence_manager: SequenceManager,
        motif_length: int,
        number_of_classes: int,
        background: BackgroundType,
    ) -> None:
        """Initialize the VariableLengthEM class.

        Args:
            sequence_manager: The instance holding the input sequences.
            motif_length: The length of the motif(s) to be estimated.
            number_of_classes: The number of motifs/classes to be identified (not counting the flat
                motif).
            background: The background amino acid frequencies. Can also be a string corresponding
                to one of the available backgrounds.
        """
        self.sm = sequence_manager

        # number of classes (excluding the flat motif)
        self.number_of_classes = number_of_classes

        # we additionally want a flat motif
        self.n_classes = number_of_classes + 1

        self.motif_length = motif_length

        self.background = Background(background)

        self._compute_similarity_weights()

    def _compute_similarity_weights(self) -> None:
        """Compute the similarity weights.

        These weights are used to downweight input sequences that share sequence with other
        sequences.
        """
        print("Total number of peptides:", len(self.sm.sequences))
        all_similarity_weights = self.sm.get_similarity_weights()
        sum_of_weights1 = np.sum(all_similarity_weights)
        print("Sum of peptide weights:", sum_of_weights1)

        self.similarity_weights = self.sm.split_array_by_size(all_similarity_weights)

    def run(
        self,
        output_directory: Openable,
        output_all_runs: bool = False,
        n_runs: int = 5,
        random_seed: int = 0,
        min_error: float = 1e-3,
        pseudocount: float = 0.1,
        force: bool = False,
    ) -> None:
        """Run the expectation-maximization algorithm.

        Args:
            output_directory: The output directory. The best model is written directly into this
                directory as well a summary of the runs. The models from the other runs are
                optionally written to subdirectories.
            output_all_runs: Whether all models obtained by the individual runs shall be returned
                (or only that of the best run).
            n_runs: Number of EM runs.
            random_seed: The random seed to be used for initializing the EM runs.
            min_error: When the log likelihood difference between two steps becomes smaller than
                this value, the EM run is finished.
            pseudocount: The pseudocounts to be used in the EM algorithm.
            force: Overwrite files if they already exist.
        """
        path = AnyPath(output_directory)

        self.score_per_run = []
        self.log_likelihood_per_run = []
        self.aic_per_run = []
        self.steps_per_run = []
        self.time_per_run = []

        # random number generator
        self.rng = np.random.default_rng(random_seed)

        self.pseudocount = pseudocount

        self.best_run = None
        self.best_score = float("-inf")

        for run in range(n_runs):
            start_time = perf_counter()

            self._initialize_new_model()

            self.runners = {}
            for length in self.sm.size_sorted_arrays.keys():
                self.runners[length] = _EqualLengthHandler(
                    self.model,
                    self.sm.size_sorted_arrays[length],
                    self.similarity_weights[length],
                )
                self.runners[length]._initialize(self.rng)

            # initialize PPMs and class weight based on initial responsibilities
            self._maximization()

            score = self._log_score()
            score_error = float("inf")

            steps = 0

            while score_error > min_error:
                steps += 1

                self._expectation()
                self._maximization()
                new_score = self._log_score()
                score_error = abs(score - new_score)
                score = new_score

                print(
                    f"Estimating frequencies (run {run:2}), "
                    f"{steps:4} EM steps, "
                    f"score = {score} ...",
                    end="\r",
                    flush=True,
                )

            self.model.is_fitted = True

            log_likelihood = self._log_likelihood()
            aic = compute_aic(self.model.num_of_parameters, log_likelihood)

            elapsed_time = perf_counter() - start_time
            print(
                f"Estimating frequencies (run {run:2}), "
                f"{steps:4} EM steps, "
                f"score = {score} ... "
                f"finished {self.sm.number_of_sequences} "
                f"sequences in {elapsed_time:.4f} seconds."
            )

            self.score_per_run.append(score)
            self.log_likelihood_per_run.append(log_likelihood)
            self.aic_per_run.append(aic)
            self.steps_per_run.append(steps)
            self.time_per_run.append(elapsed_time)

            if output_all_runs:
                path_i = path / f"run{run}"

                self.model.save(path_i, force=force)
                self._write_responsibilities(
                    path_i,
                    {length: runner.responsibilities for length, runner in self.runners.items()},
                    force=force,
                )

            if score > self.best_score:
                self.best_run = run
                self.best_score = score
                self.best_model = self.model
                self.best_responsibilities = {
                    length: runner.responsibilities for length, runner in self.runners.items()
                }

        self.write_summary(path, force=force)

    def _map_classes_to_indices(self) -> dict[str, int]:
        """Map the class names to indices.

        Raises:
            RuntimeError: If no class information is available.
            RuntimeError: If the flat motif cannot be assigned.
            RuntimeError: If there are more class names than motifs to be estimated.

        Returns:
            The mapping from class name to index.
        """
        if self.sm.classes is None:
            raise RuntimeError("class information is not available")

        class_set = sorted(set(self.sm.classes))

        mapping: dict[str, int] = {}
        flat_motif = None

        for c in class_set:
            if c.lower() in ("trash", "flat"):
                if flat_motif is not None:
                    raise RuntimeError(
                        f"flat motif could not be assigned unambiguously: '{flat_motif}' and '{c}'"
                    )
                flat_motif = c

            else:
                if len(mapping) >= self.number_of_classes:
                    raise RuntimeError("more classes than motifs to be estimated")
                mapping[c] = len(mapping)

        if flat_motif:
            mapping[flat_motif] = self.number_of_classes

        return mapping

    def run_with_known_classes(
        self, output_directory: Openable, min_error: float = 1e-3, pseudocount: float = 0.1
    ) -> None:
        """Run the expectation-maximization algorithm with known classes for initialization.

        Args:
            output_directory: The output directory. The best model is written directly into this
                directory as well a summary of the runs. The models from the other runs are
                optionally written to subdirectories.
            min_error: When the log likelihood difference between two steps becomes smaller than
                this value, the EM run is finished.
            pseudocount: The pseudocounts to be used in the EM algorithm.
        """
        self.pseudocount = pseudocount

        class_mapping = self._map_classes_to_indices()
        print("class mapping", class_mapping)

        start_time = perf_counter()

        self._initialize_new_model()

        self.runners = {}
        for length in self.sm.size_sorted_arrays.keys():
            self.runners[length] = _EqualLengthHandler(
                self.model,
                self.sm.size_sorted_arrays[length],
                self.similarity_weights[length],
            )
            self.runners[length]._initialize_with_known_classes(
                self.sm.size_sorted_classes[length], class_mapping
            )

        # initialize PPMs and class weight based on initial responsibilities
        self._maximization()

        score = self._log_score()
        score_error = float("inf")

        steps = 0

        while score_error > min_error:
            steps += 1

            self._expectation()
            self._maximization()
            new_score = self._log_score()
            score_error = abs(score - new_score)
            score = new_score

            print(
                f"Estimating frequencies (single run with known classes), "
                f"{steps:4} EM steps, "
                f" score = {score} ...",
                end="\r",
                flush=True,
            )

        self.model.is_fitted = True

        log_likelihood = self._log_likelihood()
        aic = compute_aic(self.model.num_of_parameters, log_likelihood)

        elapsed_time = perf_counter() - start_time
        print(
            f"Estimating frequencies (single run with known classes), "
            f"{steps:4} EM steps, "
            f" score = {score} ... finished {self.sm.number_of_sequences}"
            f" sequences in {elapsed_time:.4f} seconds."
        )

        self.score_per_run = [score]
        self.log_likelihood_per_run = [log_likelihood]
        self.aic_per_run = [aic]
        self.steps_per_run = [steps]
        self.time_per_run = [elapsed_time]

        self.best_run = 0
        self.best_score = score
        self.best_model = self.model
        self.best_responsibilities = {
            length: runner.responsibilities for length, runner in self.runners.items()
        }

        self.write_summary(output_directory)

    def _initialize_new_model(self) -> None:
        """Initialize a new model instance."""
        self.model = DeconvolutionModelMHC2NoOffsetWeights(
            self.sm.alphabet,
            self.motif_length,
            self.number_of_classes,
            self.background,
            has_flat_motif=True,
        )

        # position probability matrices
        self.ppm = self.model.ppm

        # the last class is the flat motif which remains unchanged
        self.ppm[self.number_of_classes] = self.background.frequencies

        # in the expectation step and for the log likehood, we do not use the raw frequencies but
        # divide them by the background frequencies, we pre-compute these "scores" after the
        # maximization step to save time
        self.pssm = self.model.pssm
        self.pssm[:] = self.ppm / self.background.frequencies

        # probalitities of the classes
        self.class_weights = self.model.class_weights

    def _log_score(self) -> float:
        """Compute the log score under the current model.

        Returns:
            Log score.
        """
        score = 0.0

        for runner in self.runners.values():
            score += runner._log_score()

        # at the moment, we use pseudocount as a constant exponent of the Dirichlet priors, hence
        # we can do the following
        score += self.pseudocount * np.sum(np.log(self.ppm[: self.number_of_classes]))

        return score

    def _log_likelihood(self) -> float:
        """Compute the log likelihood under the current model.

        Returns:
            Log likelihood.
        """
        log_likelihood = 0.0

        for runner in self.runners.values():
            log_likelihood += runner._log_likelihood()

        # at the moment, we use pseudocount as a constant exponent of the Dirichlet priors, hence
        # we can do the following
        log_likelihood += self.pseudocount * np.sum(np.log(self.ppm[: self.number_of_classes]))

        return log_likelihood

    def _maximization(self) -> None:
        """Maximization step."""
        self.ppm[: self.number_of_classes] = self.pseudocount
        self.class_weights[:] = 0

        for runner in self.runners.values():
            runner._maximization(self.ppm)

        # normalize so that frequencies sum to one for each position
        self.ppm[: self.number_of_classes] /= np.sum(
            self.ppm[: self.number_of_classes], axis=2, keepdims=True
        )

        self.pssm[: self.number_of_classes] = (
            self.ppm[: self.number_of_classes] / self.background.frequencies
        )

        # normalize so that class weights sum to one
        self.class_weights /= np.sum(self.class_weights)

    def _expectation(self) -> None:
        """Expectation step."""
        for runner in self.runners.values():
            runner._expectation()

    def get_responsibilities(self) -> dict[int, np.ndarray]:
        """The best responsibilities.

        Returns:
            The best responsibilities.
        """
        return self.best_responsibilities

    def _write_responsibilities(
        self, directory: Openable, responsibilities: dict[int, np.ndarray], force: bool
    ) -> None:
        """Write responsibility values to a file.

        Args:
            directory: The output directory.
            responsibilities: The responsibility values to be written the file
                'responsibilities.csv'.
            force: Overwrite files if they already exist.
        """
        directory = AnyPath(directory)
        n_sequences = len(self.sm.sequences)
        n_classes, _ = next(iter(responsibilities.values())).shape[1:]

        # VARIANT 1: cumulated responsibilities (over all offsets)
        cum_responsibilities = {
            length: np.sum(resp, axis=2) for length, resp in responsibilities.items()
        }

        # collect the responsibilities
        all_cum_responsibilities = np.zeros((n_sequences, n_classes))

        for i, (length, s) in enumerate(self.sm.order_in_input_file):
            all_cum_responsibilities[i, :] = cum_responsibilities[length][s, :]

        df = pd.DataFrame(
            all_cum_responsibilities,
            columns=[f"class{i+1}_cumulated" for i in range(n_classes - 1)]
            + ["flat_motif_cumulated"],
        )

        df["best_class_cumulated"] = np.argmax(all_cum_responsibilities, axis=1) + 1
        df["best_class_cumulated"] = df["best_class_cumulated"].astype(str)
        df.loc[(df["best_class_cumulated"] == n_classes), "best_class_cumulated"] = "flat"

        # VARIANT 2: maximum responsibilities directly taken from the
        # #classes x #offsets table
        best_responsibilities = []
        best_classes = []
        best_offsets = []
        binding_cores = []

        for i, (length, s) in enumerate(self.sm.order_in_input_file):
            array = responsibilities[length][s, :]
            c, o = np.unravel_index(np.argmax(array, axis=None), array.shape)
            best_responsibilities.append(array[c, o])
            if c < n_classes - 1:
                best_classes.append(c + 1)
            else:
                best_classes.append("flat")
            best_offsets.append(o)
            binding_cores.append(self.sm.sequences[i][o : o + self.model.motif_length])

        df["best_responsibility"] = best_responsibilities
        df["best_class"] = best_classes
        df["best_offset_(0-based)"] = best_offsets
        df["binding_core_prediction"] = binding_cores

        # finally insert the peptide sequence as the first column
        df.insert(0, "peptide", self.sm.sequences)

        save_csv(df, directory / "responsibilities.csv", force=force)

    def write_summary(self, directory: Openable, force: bool = False) -> None:
        """Write the best model, responsibilities, and summary of runs.

        Args:
            directory: The output directory.
            force: Overwrite files if they already exist.
        """
        directory = AnyPath(directory)
        self.best_model.save(directory, force=force)
        self._write_responsibilities(directory, self.get_responsibilities(), force)

        d = {
            "score": self.score_per_run,
            "log_likelihood": self.log_likelihood_per_run,
            "AIC": self.aic_per_run,
            "EM_steps": self.steps_per_run,
            "time": self.time_per_run,
        }
        save_csv(pd.DataFrame(d), directory / "runs.csv", force=force)
