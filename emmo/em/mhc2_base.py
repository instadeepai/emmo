"""Base class for expectation-maximization-based deconvolution of MHC2 ligands."""
from __future__ import annotations

from abc import abstractmethod

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from emmo.em.base_runner import BaseRunner
from emmo.io.file import Openable
from emmo.io.file import save_csv
from emmo.models.deconvolution import DeconvolutionModelMHC2
from emmo.pipeline.background import Background
from emmo.pipeline.background import BackgroundType
from emmo.pipeline.sequences import SequenceManager
from emmo.utils import logger
from emmo.utils.offsets import AlignedOffsets
from emmo.utils.statistics import compute_aic

log = logger.get(__name__)


class BaseEMRunnerMHC2(BaseRunner):
    """Base class for running MHC2 expectation maximization algorithms."""

    def __init__(
        self,
        sequence_manager: SequenceManager,
        motif_length: int,
        number_of_classes: int,
        background: BackgroundType,
    ) -> None:
        """Initialize the MHC2 EM runner base class.

        Args:
            sequence_manager: The instance holding the input sequences.
            motif_length: The length of the motif(s) to be estimated.
            number_of_classes: The number of motifs/classes to be identified (not counting the flat
                motif).
            background: The background amino acid frequencies. Can also be a string corresponding to
                one of the available backgrounds.

        Raises:
            ValueError: If the minimal sequence length is shorter than the specified motif length.
        """
        super().__init__(sequence_manager, motif_length, number_of_classes)

        self.best_model: DeconvolutionModelMHC2 | None = None
        self.current_pssm: np.ndarray | None = None
        self.current_class_weights: np.ndarray | None = None
        self.current_log_likelihood_ppm: float = float("-inf")

        if self.sm.get_minimal_length() < self.motif_length:
            raise ValueError(
                f"input contains sequences that are shorter than motif length {self.motif_length}"
            )

        # auxiliary class for aligning offsets
        self.aligned_offsets = AlignedOffsets(self.motif_length, self.sm.get_maximal_length())
        self.n_offsets = self.aligned_offsets.get_number_of_offsets()

        # background amino acid distribution
        self.background = Background(background)
        background_name = self.background.name if self.background.name else "custom"
        log.info(f"Using background frequencies: {background_name}")

        # compute similarity weights and correction factor for the middle offset
        self._compute_similarity_weights()
        log.info(f"Effective number of sequences: {self.sum_of_weights:4f}")
        log.info(f"Upweighting middle offset by factor {self.upweight_middle_offset:4f}")

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

    @abstractmethod
    def _expectation_maximization(self) -> None:
        """Do one expectation-maximization run.

        Does one EM run until convergence and sets at least the following attributes:
        - self.current_class_weights
        - self.current_ppm
        - self.current_score
        - self.current_responsibilities
        - self.current_steps
        - self.current_pssm
        - self.current_log_likelihood_ppm
        """

    def _runner_specific_run_details(self, num_of_parameters: int) -> None:
        """Add the runner-specific information to the 'run_details' dictionary.

        Args:
            num_of_parameters: Number of model parameters.
        """
        self.run_details["score"].append(self.current_score)
        self.run_details["log_likelihood"].append(self.current_log_likelihood_ppm)
        self.run_details["AIC_PPM"].append(
            compute_aic(num_of_parameters, self.current_log_likelihood_ppm)
        )
        self.run_details["AIC_PSSM"].append(compute_aic(num_of_parameters, self.current_score))

    def _build_model(self) -> DeconvolutionModelMHC2:
        """Collect current PPM and class weights into a model.

        Returns:
            The current parameters (PPMs, class weights, etc.) collected in an instance of the
            corresponding deconvolution model class.
        """
        model = DeconvolutionModelMHC2(
            self.sm.alphabet,
            self.motif_length,
            self.number_of_classes,
            self.sm.get_maximal_length(),
            self.background,
            has_flat_motif=True,
        )

        model.ppm = self.current_ppm
        model.pssm = self.current_pssm
        model.class_weights = self.current_class_weights
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
