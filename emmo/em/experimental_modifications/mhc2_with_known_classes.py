"""Implementation of the EM algorithm for MHC2 ligands with given class initialization.

Here the class information must be provided in the input file and the responsibilities are not
initialized at random.
"""
from __future__ import annotations

import numpy as np

from emmo.em.mhc2 import EMRunnerMHC2
from emmo.io.file import Openable
from emmo.io.sequences import SequenceManager


class EMRunnerMHC2KnownClasses(EMRunnerMHC2):
    """Class for running the EM algorithm for MHC2 ligands with given class initialization."""

    def __init__(
        self,
        sequence_manager: SequenceManager,
        motif_length: int,
        number_of_classes: int,
        background: str = "MHC2_biondeep",
    ) -> None:
        """Initialize the MHC2 EM runner base class with given class initialization.

        The provided 'sequence_manager' must contain valid class information.

        Args:
            sequence_manager: The instance holding the input sequences.
            motif_length: The length of the motif(s) to be estimated.
            number_of_classes: The number of motifs/classes to be identified (not counting the flat
                motif).
            background: The background amino acid frequencies. Must be a string corresponding to
                one of the available backgrounds.

        Raises:
            ValueError: If the minimal sequence length is shorter than the specified motif length.
            RuntimeError: If no class information is available.
        """
        super().__init__(
            sequence_manager,
            motif_length,
            number_of_classes,
            background=background,
        )

        if self.sm.classes is None:
            raise RuntimeError("class information is not available")
        self.classes: list[str] = self.sm.classes

        self.class_mapping = self._map_classes_to_indices()

    def _initialize_responsibilities(self) -> None:
        """Initialize the responsibilities with known classes.

        Returns:
            The initialized responsibilities.
        """
        self.responsibilities = np.zeros((self.n_sequences, self.n_classes, self.n_offsets))

        for s in range(self.n_sequences):
            c = self.class_mapping[self.classes[s]]
            self.responsibilities[s, c, :] = 1

        # normalize the initial responsibilities
        self.responsibilities /= np.sum(self.responsibilities, axis=(1, 2), keepdims=True)

    def _map_classes_to_indices(self) -> dict[str, int]:
        """Map the class names to indices.

        Raises:
            RuntimeError: If the flat motif cannot be assigned.
            RuntimeError: If there are more class names than motifs to be estimated.

        Returns:
            The mapping from class name to index.
        """
        class_set = sorted(set(self.classes))

        mapping: dict[str, int] = {}
        flat_motif = None

        for c in class_set:
            if c.lower() in ("trash", "flat"):
                if flat_motif is not None:
                    raise RuntimeError(
                        f"flat motif could not be assigned unambiguously: "
                        f"'{flat_motif}' and '{c}'"
                    )
                flat_motif = c

            else:
                if len(mapping) >= self.number_of_classes:
                    raise RuntimeError("more classes than motifs to be estimated")
                mapping[c] = len(mapping)

        if flat_motif:
            mapping[flat_motif] = self.number_of_classes

        return mapping

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
        # when the initial classes are given, we just need one EM run as they will produce
        # identical results
        output_all_runs = False
        n_runs = 1

        print(
            f"--------------------------------------------------------------\n"
            f"Running EM algorithm with {self.number_of_classes} classes\n"
            f"with given class initialization --> just doing 1 EM run\n"
        )

        super().run(
            output_directory,
            output_all_runs=output_all_runs,
            n_runs=n_runs,
            random_seed=random_seed,
            min_error=min_error,
            pseudocount=pseudocount,
            force=force,
        )
