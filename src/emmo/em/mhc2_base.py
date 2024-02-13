"""Base class for expectation-maximization-based deconvolution of MHC2 ligands."""
from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from emmo.io.file import Openable
from emmo.io.file import save_csv
from emmo.io.sequences import SequenceManager
from emmo.models.deconvolution import DeconvolutionModelMHC2
from emmo.resources.background_freqs import get_background
from emmo.utils.offsets import AlignedOffsets
from emmo.utils.statistics import compute_aic


class BaseEMRunnerMHC2:
    """Base class for running MHC2 expectation maximization algorithms."""

    def __init__(
        self,
        sequence_manager: SequenceManager,
        motif_length: int,
        number_of_classes: int,
        background: str = "MHC2_biondeep",
    ) -> None:
        """Initialize the MHC2 EM runner base class.

        Args:
            sequence_manager: The instance holding the input sequences.
            motif_length: The length of the motif(s) to be estimated.
            number_of_classes: The number of motifs/classes to be identified (not counting the flat
                motif).
            background: The background amino acid frequencies. Must be a string corresponding to
                one of the available backgrounds.

        Raises:
            ValueError: If the minimal sequence length is shorter than the specified motif length.
        """
        self.sm = sequence_manager

        self.sequences = self.sm.sequences_as_indices()
        self.n_sequences = len(self.sequences)

        # number of classes (excluding the flat motif)
        self.number_of_classes = number_of_classes

        # we additionally want a flat motif
        self.n_classes = number_of_classes + 1

        self.motif_length = motif_length
        self.n_alphabet = len(self.sm.alphabet)

        if self.sm.get_minimal_length() < self.motif_length:
            raise ValueError(
                f"input contains sequences that are shorter than "
                f"motif length {self.motif_length}"
            )

        # auxiliary class for aligning offsets
        self.aligned_offsets = AlignedOffsets(self.motif_length, self.sm.get_maximal_length())
        self.n_offsets = self.aligned_offsets.get_number_of_offsets()

        # background amino acid distribution
        self.background = background
        self.background_freqs = get_background(which=background)

        self._compute_similarity_weights()

    def _compute_similarity_weights(self) -> None:
        """Compute the similarity weights.

        These weights are used to downweight input sequences that share sequence with other
        sequences.
        """
        self.similarity_weights = self.sm.get_similarity_weights()
        self.sum_of_weights = np.sum(self.similarity_weights)

        self.similarity_weights_by_length = self.sm.split_array_by_size(self.similarity_weights)

        # as in MoDec, the offset position "0" is skipped for all sequences where
        # (seq. length - motif length) % 2 != 0;
        # therefore, we upweight the weights for this middle offset here
        sum_of_even = sum(
            np.sum(w)
            for length, w in self.similarity_weights_by_length.items()
            if (length - self.motif_length) % 2 == 0
        )
        self.upweight_middle_offset = self.sum_of_weights / sum_of_even

        self.offset_upweighting = np.full(shape=(self.n_offsets,), fill_value=1, dtype=np.float64)
        self.offset_upweighting[self.n_offsets // 2] = self.upweight_middle_offset

    def _initialize_responsibilities(self) -> np.ndarray:
        """Initialize the responsibilities at random.

        The responsibilities are initialized by assigning every sequence to a class at random while
        ensuring that, if enough sequences are available, every class has at least one sequence
        assigned to it. The sequence is assigned to all possible offsets with uniform
        responsibility.

        Returns:
            The initialized responsibilities.
        """
        # initialize responsibilities by assigning a random class to each sequence
        _responsibilities = np.zeros((self.n_sequences, self.n_classes, self.n_offsets))

        random_classes = self.rng.integers(self.n_classes, size=self.n_sequences, dtype=np.uint32)

        # ensure that at least one sequence is assigned to each class
        for c, s in enumerate(
            self.rng.choice(
                self.n_sequences,
                size=min(self.n_sequences, self.n_classes),
                replace=False,
                shuffle=True,
            )
        ):
            random_classes[s] = c

        # here we use uniform responsibility for all possible offsets
        for s in range(self.n_sequences):
            for o in self.aligned_offsets.get_offset_list(len(self.sequences[s])):
                _responsibilities[s, random_classes[s], o] = 1

        # normalize the initial responsibilities
        _responsibilities /= np.sum(_responsibilities, axis=(1, 2), keepdims=True)

        return _responsibilities

    def _expectation_maximization(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, np.ndarray, int,]:
        """Do one expectation-maximization run.

        Raises:
            NotImplementedError: If the method is called for the base class or the child class does
                not override this method.

        Returns:
            The class weights, PPM, PSSM, log likelihood (using PSSM), log likelihood (using PPM),
            responsibilities, and EM steps.
        """
        raise NotImplementedError(
            "Expectation-maximzation algorithm must be implemented by the child classes"
        )

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
        print(
            f"--------------------------------------------------------------\n"
            f"Running EM algorithm with {self.number_of_classes} classes\n"
            f"--------------------------------------------------------------\n"
            f"Total number of peptides: {len(self.sm.sequences)}\n"
            f"Effective number of peptides (sum of weights): "
            f"{self.sum_of_weights}\n"
            f"Upweighting middle offset by factor: "
            f"{self.upweight_middle_offset}\n"
            f"Number of runs: {n_runs}\n"
            f"Log likelihood difference for stopping: {min_error}\n"
            f"Random seed: {random_seed}\n"
            f"Pseudocount: {pseudocount}\n"
            f"--------------------------------------------------------------"
        )

        self.n_runs = n_runs
        self.random_seed = random_seed
        self.min_error = min_error
        self.pseudocount = pseudocount

        path = AnyPath(output_directory)

        self.run_details: dict[str, list[Any]] = {
            "score": [],
            "log_likelihood": [],
            "AIC_PPM": [],
            "AIC_PSSM": [],
            "EM_steps": [],
            "time": [],
        }

        # random number generator
        self.rng = np.random.default_rng(self.random_seed)

        self.best_run = None
        self.best_score = float("-inf")

        for run in range(self.n_runs):
            self.current_run = run
            start_time = perf_counter()

            (
                self.class_weights,
                self.ppm,
                self.pssm,
                log_likelihood_pssm,
                log_likelihood_ppm,
                self.responsibilities,
                steps,
            ) = self._expectation_maximization()

            model = self._current_state_to_model()
            elapsed_time = perf_counter() - start_time

            print(
                f"Estimating frequencies (run {run:2}), "
                f"{steps:4} EM steps, "
                f"score = {log_likelihood_pssm} ... "
                f"finished {self.sm.number_of_sequences()} "
                f"sequences in {elapsed_time:.4f} seconds."
            )

            self.run_details["score"].append(log_likelihood_pssm)
            self.run_details["log_likelihood"].append(log_likelihood_ppm)
            self.run_details["AIC_PPM"].append(
                compute_aic(model.get_number_of_parameters(), log_likelihood_ppm)
            )
            self.run_details["AIC_PSSM"].append(
                compute_aic(model.get_number_of_parameters(), log_likelihood_pssm)
            )
            self.run_details["EM_steps"].append(steps)
            self.run_details["time"].append(elapsed_time)

            if output_all_runs:
                path_i = path / f"run{run}"
                model.save(path_i, force=force)
                self._write_responsibilities(path_i, self.responsibilities, force=force)

            if log_likelihood_pssm > self.best_score:
                self.best_run = run
                self.best_score = log_likelihood_pssm
                self.best_model = model
                self.best_responsibilities = self.responsibilities

        self.write_summary(path, force=force)

    def _current_state_to_model(self) -> DeconvolutionModelMHC2:
        """Collect current PPM and class weights into a model.

        Returns:
            The current state as a model.
        """
        model = DeconvolutionModelMHC2(
            self.sm.alphabet,
            self.motif_length,
            self.number_of_classes,
            self.sm.get_maximal_length(),
            has_flat_motif=True,
            background=self.background,
        )

        model.ppm = self.ppm
        model.pssm = self.pssm
        model.class_weights = self.class_weights
        model.is_fitted = True

        model.training_params["total_peptide_number"] = len(self.sm.sequences)
        model.training_params["effective_peptide_number"] = self.sum_of_weights
        model.training_params["run_number"] = self.current_run
        model.training_params["total_run_number"] = self.n_runs
        model.training_params["random_seed"] = self.random_seed
        model.training_params["min_log_likelihood_error"] = self.min_error
        model.training_params["pseudocount"] = self.pseudocount

        return model

    def _write_responsibilities(
        self,
        directory: Openable,
        responsibilities: np.ndarray,
        force: bool,
    ) -> None:
        """Write responsibility values to a file.

        Args:
            directory: The output directory.
            responsibilities: The responsibility values to be written the file
                'responsibilities.csv'.
            force: Overwrite files if they already exist.
        """
        directory = AnyPath(directory)

        # VARIANT 1: cumulated responsibilities (over all offsets)
        cum_responsibilities = np.sum(responsibilities, axis=2)

        df = pd.DataFrame(
            cum_responsibilities,
            columns=[f"class{i+1}_cumulated" for i in range(self.n_classes - 1)]
            + ["flat_motif_cumulated"],
        )

        df["best_class_cumulated"] = np.argmax(cum_responsibilities, axis=1) + 1
        df["best_class_cumulated"] = df["best_class_cumulated"].astype(str)
        df.loc[(df["best_class_cumulated"] == str(self.n_classes)), "best_class_cumulated"] = "flat"

        # VARIANT 2: maximum responsibilities directly taken from the #classes x #offsets table
        best_responsibilities = []
        best_classes = []
        best_offsets = []
        binding_cores = []

        for s in range(self.n_sequences):
            array = responsibilities[s]
            length = len(self.sequences[s])

            c, o = np.unravel_index(np.argmax(array, axis=None), array.shape)
            best_responsibilities.append(array[c, o])

            if c < self.n_classes - 1:
                best_classes.append(c + 1)
            else:
                best_classes.append("flat")

            offset = self.aligned_offsets.get_offset_in_sequence(length, o)
            best_offsets.append(offset)
            binding_cores.append(self.sm.sequences[s][offset : offset + self.motif_length])

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
        self.best_model.save(directory, force=force)
        self._write_responsibilities(directory, self.best_responsibilities, force=force)
        save_csv(pd.DataFrame(self.run_details), AnyPath(directory) / "runs.csv", force=force)
