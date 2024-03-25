"""Base class for expectation-maximization-based deconvolution of MHC1 ligands."""
from __future__ import annotations

from abc import abstractmethod

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from emmo.em.base_runner import BaseRunner
from emmo.io.file import Openable
from emmo.io.file import save_csv
from emmo.models.deconvolution import DeconvolutionModelMHC1
from emmo.pipeline.sequences import SequenceManager
from emmo.utils.statistics import compute_aic


class BaseEMRunnerMHC1(BaseRunner):
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
        super().__init__(sequence_manager, motif_length, number_of_classes)

        self.n_term = n_term
        self.c_term = c_term
        self.n_term_penalty = n_term_penalty
        self.c_term_penalty = c_term_penalty

        self.best_model: DeconvolutionModelMHC1 | None = None
        self.current_ppm: np.ndarray | None = None
        self.current_class_weights: dict[int, np.ndarray] | None = None

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

    @abstractmethod
    def _expectation_maximization(self) -> None:
        """Do one expectation-maximization run.

        Does one EM run until convergence and sets the following attributes:
        - self.current_class_weights
        - self.current_ppm
        - self.current_score
        - self.current_responsibilities
        - self.current_steps
        """

    def _runner_specific_run_details(self, num_of_parameters: int) -> None:
        """Add the runner-specific information to the 'run_details' dictionary.

        Args:
            num_of_parameters: Number of model parameters.
        """
        self.run_details["log_likelihood"].append(self.current_score)
        self.run_details["AIC"].append(compute_aic(num_of_parameters, self.current_score))

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

        model.ppm = self.current_ppm
        model.class_weights = self.current_class_weights
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
