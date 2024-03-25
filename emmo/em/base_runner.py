"""Base class for expectation-maximization-based deconvolution."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from collections import defaultdict
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from emmo.io.file import Openable
from emmo.io.file import save_csv
from emmo.models.deconvolution import DeconvolutionModel
from emmo.pipeline.sequences import SequenceManager
from emmo.utils import logger

log = logger.get(__name__)


class BaseRunner(ABC):
    """Base class for running expectation maximization algorithms."""

    def __init__(
        self,
        sequence_manager: SequenceManager,
        motif_length: int,
        number_of_classes: int,
    ) -> None:
        """Initialize the base class.

        Args:
            sequence_manager: The instance holding the input sequences.
            motif_length: The length of the motif(s) to be estimated.
            number_of_classes: The number of motifs to be identified (not counting the flat motif).
        """
        log.info(
            f"Initializing {type(self).__name__} (motif length {motif_length}, "
            f"{number_of_classes} class{'es' if number_of_classes != 1 else ''})"
        )

        self.sm = sequence_manager
        self.sequences = self.sm.sequences_as_indices()
        self.n_sequences = len(self.sequences)
        log.info(f"Total number of sequences: {self.n_sequences}")

        # number of classes (excluding the flat motif)
        self.number_of_classes = number_of_classes

        # we additionally want a flat motif
        self.n_classes = number_of_classes + 1

        self.motif_length = motif_length
        self.n_alphabet = len(self.sm.alphabet)

        self.best_model: DeconvolutionModel | None = None
        self.best_responsibilities: np.ndarray | None = None
        self.best_run: int | None = None
        self.best_score: float | None = float("-inf")
        self.run_details: dict[str, list[Any]] = defaultdict(list)

        self.current_steps: int | None = None
        self.current_responsibilities: np.ndarray | None = None
        self.current_run: int | None = None
        self.current_score: float | None = float("-inf")
        self.current_ppm: np.ndarray | None = None

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
        log.info(
            f"Starting EM algorithm ({self.number_of_classes} "
            f"class{'es' if self.number_of_classes != 1 else ''}, {n_runs} "
            f"run{'s' if n_runs != 1 else ''})"
        )
        log.info(
            f"Using random seed {random_seed}, pseudocount {pseudocount}, "
            f"delta LL for stopping {min_error}"
        )

        self.n_runs = n_runs
        self.random_seed = random_seed
        self.min_error = min_error
        self.pseudocount = pseudocount

        path = AnyPath(output_directory)

        # (re)set variables storing for the best model
        self.best_model = None
        self.best_responsibilities = None
        self.best_run = None
        self.best_score = float("-inf")
        self.run_details = defaultdict(list)

        # random number generator
        self.rng = np.random.default_rng(self.random_seed)

        for run in range(1, self.n_runs + 1):
            log.info(
                f"{self.number_of_classes} class{'es' if self.number_of_classes != 1 else ''}, "
                f"run {run}/{self.n_runs} started"
            )

            self.current_run = run

            start_time = perf_counter()
            self._expectation_maximization()
            elapsed_time = perf_counter() - start_time

            log.info(
                f"{self.number_of_classes} class{'es' if self.number_of_classes != 1 else ''}, "
                f"run {run}/{self.n_runs} done "
                f"({self.current_steps} steps, {elapsed_time:.4f} s, score = {self.current_score})"
            )

            model = self._current_state_to_model()
            self.run_details["EM_steps"].append(self.current_steps)
            self.run_details["time"].append(elapsed_time)
            self._runner_specific_run_details(model.num_of_parameters)

            if output_all_runs:
                path_i = path / f"run{run}"
                model.save(path_i, force=force)
                self._write_responsibilities(path_i, self.current_responsibilities, force=force)

            if self.current_score > self.best_score:
                self.best_run = run
                self.best_score = self.current_score
                self.best_model = model
                self.best_responsibilities = self.current_responsibilities

        log.info(
            f"Finished {n_runs} run{'s' if n_runs != 1 else ''} "
            f"(best run: {self.best_run}, score = {self.best_score})"
        )

        self.write_summary(path, force=force)

    def write_summary(self, directory: Openable, force: bool = False) -> None:
        """Write the best model, responsibilities, and summary of runs.

        Args:
            directory: The output directory.
            force: Overwrite files if they already exist.
        """
        if self.best_model is None:
            raise ValueError("call method 'run' before writing the summary")

        self.best_model.save(directory, force=force)
        self._write_responsibilities(directory, self.best_responsibilities, force)
        save_csv(pd.DataFrame(self.run_details), AnyPath(directory) / "runs.csv", force=force)

    @abstractmethod
    def _expectation_maximization(self) -> None:
        """Do one expectation-maximization run.

        Does one EM run until convergence and sets at least the following attributes:
        - self.current_class_weights
        - self.current_ppm
        - self.current_score
        - self.current_responsibilities
        - self.current_steps
        """

    @abstractmethod
    def _current_state_to_model(self) -> DeconvolutionModel:
        """Collect current PPM and class weights into a model.

        Returns:
            The current state as a model.
        """

    @abstractmethod
    def _runner_specific_run_details(self, num_of_parameters: int) -> None:
        """Add the runner-specific information to the 'run_details' dictionary.

        Args:
            num_of_parameters: Number of model parameters.
        """

    @abstractmethod
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
