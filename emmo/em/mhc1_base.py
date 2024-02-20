"""Base class for expectation-maximization-based deconvolution of MHC1 ligands."""
from __future__ import annotations

from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from emmo.io.file import Openable
from emmo.io.file import save_csv
from emmo.io.sequences import SequenceManager
from emmo.models.deconvolution import DeconvolutionModelMHC1
from emmo.utils.statistics import compute_aic


class BaseEMRunnerMHC1:
    """Base class for running the deconvolution of MHC1 ligands."""

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
        """Initialize the base class.

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
        self.sm = sequence_manager

        self.sequences = self.sm.sequences_as_indices()
        self.n_sequences = len(self.sequences)

        # number of classes (excluding the flat motif)
        self.number_of_classes = number_of_classes

        # we additionally want a flat motif
        self.n_classes = number_of_classes + 1

        self.n_alphabet = len(self.sm.alphabet)
        self.motif_length = motif_length

        self.n_term = n_term
        self.c_term = c_term
        self.n_term_penalty = n_term_penalty
        self.c_term_penalty = c_term_penalty

    def _initialize_responsibilities(self) -> np.ndarray:
        """Iniliatialize the responsibilities at random for all sequences.

        Returns:
            The initialized responsibilities.
        """
        # initialize responsibilities by assigning a random class to each sequence
        _responsibilities = np.zeros((self.n_sequences, self.n_classes))

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
            _responsibilities[i, random_classes[i]] = 1

        return _responsibilities

    def _expectation_maximization(
        self,
    ) -> tuple[dict[int, np.ndarray], np.ndarray, float, np.ndarray, int,]:
        """Do one expectation-maximization run.

        Returns:
            The class weights, PPM, log likelihood, responsibilities, and EM steps.

        Raises:
            NotImplementedError: If the method is called for the base class or the child class does
                not override this method.
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
            "log_likelihood": [],
            "AIC": [],
            "EM_steps": [],
            "time": [],
        }

        # random number generator
        self.rng = np.random.default_rng(self.random_seed)

        self.best_run = None
        self.best_log_likelihood = float("-inf")

        for run in range(self.n_runs):
            self.current_run = run
            start_time = perf_counter()

            (
                self.class_weights,
                self.ppm,
                log_likelihood,
                self.responsibilities,
                steps,
            ) = self._expectation_maximization()

            model = self._current_state_to_model()
            elapsed_time = perf_counter() - start_time

            print(
                f"Estimating frequencies (run {run:2}), "
                f"{steps:4} EM steps, "
                f" logL = {log_likelihood} ... "
                f"finished {self.sm.number_of_sequences()} "
                f"sequences in {perf_counter()-start_time:.4f} seconds."
            )

            self.run_details["log_likelihood"].append(log_likelihood)
            self.run_details["AIC"].append(
                compute_aic(model.get_number_of_parameters(), log_likelihood)
            )
            self.run_details["EM_steps"].append(steps)
            self.run_details["time"].append(elapsed_time)

            if output_all_runs:
                path_i = path / f"run{run}"
                model.save(path_i, force=force)
                self._write_responsibilities(path_i, self.responsibilities, force=force)

            if log_likelihood > self.best_log_likelihood:
                self.best_run = run
                self.best_log_likelihood = log_likelihood
                self.best_model = model
                self.best_responsibilities = self.responsibilities

        self.write_summary(path, force=force)

    def _current_state_to_model(self) -> DeconvolutionModelMHC1:
        """Collect current PPM and class weights into a model.

        Returns:
            The current state as a model.
        """
        model = DeconvolutionModelMHC1(
            self.sm.alphabet,
            self.motif_length,
            self.number_of_classes,
            has_flat_motif=True,
        )

        model.ppm = self.ppm
        model.class_weights = self.class_weights
        model.is_fitted = True

        model.training_params["total_peptide_number"] = len(self.sm.sequences)
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

        df = pd.DataFrame(
            responsibilities,
            columns=[f"class{i+1}" for i in range(self.n_classes - 1)] + ["flat_motif"],
        )

        df["best_class"] = np.argmax(responsibilities, axis=1) + 1
        df["best_class"] = df["best_class"].astype(str)
        df.loc[(df["best_class"] == str(self.n_classes)), "best_class"] = "flat"

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
        self._write_responsibilities(directory, self.best_responsibilities, force)
        save_csv(pd.DataFrame(self.run_details), AnyPath(directory) / "runs.csv", force=force)
