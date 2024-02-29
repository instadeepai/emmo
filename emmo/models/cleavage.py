"""Module for th cleavage model."""
from __future__ import annotations

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from emmo.io.file import load_csv
from emmo.io.file import load_json
from emmo.io.file import Openable
from emmo.io.file import save_csv
from emmo.io.file import save_json
from emmo.io.output import write_matrices
from emmo.pipeline.background import Background
from emmo.pipeline.background import BackgroundType
from emmo.utils.exceptions import NotFittedError


class CleavageModel:
    """Model to capture presentation-related signals at the N- and C-terminus of peptides.

    This is e.g. intended to capture peptide cleavage related signals to improve presentation
    prediction.
    """

    def __init__(
        self,
        alphabet: str | tuple[str, ...] | list[str],
        number_of_classes: int,
        background: BackgroundType,
        n_terminus_length: int = 3,
        c_terminus_length: int = 3,
        has_flat_motif: bool = True,
    ) -> None:
        """Initialize the cleavage model.

        Args:
            alphabet: The (amino acid) alphabet.
            number_of_classes: Number of motifs/classes (excl. the flat motif). Must be the same
                for N- and C-terminus.
            background: The background amino acid frequencies. Can also be a string corresponding
                to one of the available backgrounds.
            n_terminus_length: The length of the motif for the N-terminus.
            c_terminus_length: The length of the motif for the C-terminus.
            has_flat_motif: Whether to include flat motifs.
        """
        self.alphabet = alphabet
        self.n_alphabet = len(alphabet)
        self.alphabet_index = {a: i for i, a in enumerate(alphabet)}

        self.background = Background(background)

        self.n_terminus_length = n_terminus_length
        self.c_terminus_length = c_terminus_length

        self.number_of_classes = number_of_classes
        self.n_classes = number_of_classes + (1 if has_flat_motif else 0)

        self.has_flat_motif = has_flat_motif

        self.is_fitted = False

        self._initialize_arrays()

    @classmethod
    def load(cls, directory: Openable) -> CleavageModel:
        """Load a model from a directory.

        Args:
            directory: The directory containing the model files.

        Returns:
            The loaded cleavage model.
        """
        directory = AnyPath(directory)

        model_specs = load_json(directory / "model_specs.json")

        model = cls(
            model_specs["alphabet"],
            model_specs["number_of_classes"],
            model_specs["background"],
            n_terminus_length=model_specs["n_terminus_length"],
            c_terminus_length=model_specs["c_terminus_length"],
            has_flat_motif=model_specs["has_flat_motif"],
        )

        n = model_specs["number_of_classes"]

        for subfolder, terminus_length, class_weights, ppm in zip(
            ["N-terminus", "C-terminus"],
            [model.n_terminus_length, model.c_terminus_length],
            [model.class_weights_n, model.class_weights_c],
            [model.ppm_n, model.ppm_c],
        ):
            # class weights
            matrix = load_csv(
                directory / subfolder / f"class_weights_{n}.csv", index_col=0
            ).to_numpy()

            if matrix.shape != (model.n_classes, 1):
                raise ValueError(
                    "class weights file is invalid: "
                    f"required shape {(model.n_classes, )} but got {matrix.shape}"
                )

            class_weights[:] = matrix[:, 0]

            # PPMs
            for i in range(n):
                file = directory / subfolder / f"matrix_{n}_{i+1}.csv"
                matrix = load_csv(file, index_col=0, header=0).to_numpy()
                if matrix.shape != (terminus_length, model.n_alphabet):
                    raise ValueError(f"invalid frequency matrix file '{file}'")
                ppm[i] = matrix

            if model.has_flat_motif:
                matrix = load_csv(
                    directory / subfolder / f"matrix_{n}_flat.csv", index_col=0, header=0
                ).to_numpy()
                ppm[-1] = matrix[0]

        model.recompute_pssm()
        model.is_fitted = True

        return model

    @classmethod
    def load_from_separate_models(
        cls,
        directory_n: Openable,
        directory_c: Openable,
    ) -> CleavageModel:
        """Compile a model from separate deconvolution model for N- and C-terminus.

        Args:
            directory_n: Directory of the model for the N-terminus.
            directory_c: Directory of the model for the C-terminus.

        Returns:
            The compiled cleavage model.
        """
        directory_n = AnyPath(directory_n)
        directory_c = AnyPath(directory_c)

        model_specs_n = load_json(directory_n / "model_specs.json")
        model_specs_c = load_json(directory_c / "model_specs.json")

        for spec in ("alphabet", "number_of_classes", "has_flat_motif", "background"):
            if model_specs_n[spec] != model_specs_c[spec]:
                raise ValueError(f"{spec} differs between N- and C-terminus models")

        model = cls(
            model_specs_n["alphabet"],
            model_specs_n["number_of_classes"],
            model_specs_n["background"],
            n_terminus_length=model_specs_n["motif_length"],
            c_terminus_length=model_specs_c["motif_length"],
            has_flat_motif=model_specs_n["has_flat_motif"],
        )

        n = model.number_of_classes

        for directory, terminus_length, class_weights, ppm in zip(
            [directory_n, directory_c],
            [model.n_terminus_length, model.c_terminus_length],
            [model.class_weights_n, model.class_weights_c],
            [model.ppm_n, model.ppm_c],
        ):
            # class weights
            matrix = load_csv(directory / f"class_weights_{n}.csv", index_col=0).to_numpy()

            if matrix.shape != (model.n_classes, 1):
                raise ValueError(
                    "class weights file is invalid: required shape "
                    f"{(model.n_classes, )} but got {matrix.shape}"
                )

            class_weights[:] = matrix[:, 0]

            # PPMs
            for i in range(n):
                file = directory / f"matrix_{n}_{i+1}.csv"
                matrix = load_csv(file, index_col=0, header=0).to_numpy()
                if matrix.shape != (terminus_length, model.n_alphabet):
                    raise ValueError(f"invalid frequency matrix file '{file}'")
                ppm[i] = matrix

            if model.has_flat_motif:
                matrix = load_csv(
                    directory / f"matrix_{n}_flat.csv", index_col=0, header=0
                ).to_numpy()
                ppm[-1] = matrix[0]

        model.recompute_pssm()
        model.is_fitted = True

        return model

    def save(self, directory: Openable, force: bool = False) -> None:
        """Save the model to a directory.

        Args:
            directory: The directory where to save the model.
            force: Overwrite existing model files.

        Raises:
            NotFittedError: If the model has not yet been fitted.
        """
        if not self.is_fitted:
            raise NotFittedError()

        directory = AnyPath(directory)
        directory_n = directory / "N-terminus"
        directory_c = directory / "C-terminus"

        model_specs = {
            x: self.__dict__[x]
            for x in [
                "alphabet",
                "number_of_classes",
                "n_terminus_length",
                "c_terminus_length",
                "has_flat_motif",
            ]
        }
        model_specs["background"] = self.background.get_representation()

        save_json(model_specs, directory / "model_specs.json", force=force)

        # N-terminus
        write_matrices(
            directory_n,
            self.ppm_n[:-1],
            self.alphabet,
            flat_motif=self.ppm_n[-1, 0, :],
            force=force,
        )

        save_csv(
            pd.DataFrame(self.class_weights_n, index=list(range(1, self.n_classes)) + ["flat"]),
            directory_n / f"class_weights_{self.n_classes-1}.csv",
            force=force,
        )

        # C-terminus
        write_matrices(
            directory_c,
            self.ppm_c[:-1],
            self.alphabet,
            flat_motif=self.ppm_c[-1, 0, :],
            force=force,
        )

        save_csv(
            pd.DataFrame(self.class_weights_c, index=list(range(1, self.n_classes)) + ["flat"]),
            directory_c / f"class_weights_{self.n_classes-1}.csv",
            force=force,
        )

    def _initialize_arrays(self) -> None:
        """Initialize arrays.

        Initializes arrays for the position probability and scoring matrices, as well as for the
        class weights.
        """
        # position probability matrices for N-terminus and C-terminus
        self.ppm_n = np.zeros((self.n_classes, self.n_terminus_length, self.n_alphabet))
        self.ppm_c = np.zeros((self.n_classes, self.c_terminus_length, self.n_alphabet))

        # position-specific scoring matrix
        self.pssm_n = np.zeros_like(self.ppm_n)
        self.pssm_c = np.zeros_like(self.ppm_c)

        # probalitities of the classes
        self.class_weights_n = np.zeros(self.n_classes)
        self.class_weights_c = np.zeros(self.n_classes)

    def recompute_pssm(self) -> None:
        """Update the PSSMs using the current PPMs and the background."""
        self.pssm_n[:] = self.ppm_n / self.background.frequencies
        self.pssm_c[:] = self.ppm_c / self.background.frequencies

    def get_number_of_parameters(self) -> int:
        """Return the number of parameters (frequencies and class priors).

        Returns:
            The number of parameters.
        """
        # class weight incl. flat motif (subtract 1 because weights sum to one, times 2 for N- and
        # C-terminus)
        count = 2 * (self.n_classes - 1)

        # frequencies summed over all other classes (subtract 1 because aa frequencies sum to one)
        count += (
            self.number_of_classes
            * (self.n_terminus_length + self.c_terminus_length)
            * (self.n_alphabet - 1)
        )

        return count

    def score_peptide(self, peptide: str, include_flat: bool = True) -> float:
        """Score a peptide with this model.

        Use the PSSMs for the N- and C-terminus.

        Args:
            peptide: The peptide to be scored.
            include_flat: Whether to include the flat motif in the sum of scores.

        Returns:
            The score.
        """
        # TODO: think about how to handle such peptides
        if len(peptide) < self.n_terminus_length + self.c_terminus_length:
            return 0.0

        classes = self.n_classes
        if not include_flat and self.has_flat_motif:
            classes -= 1

        # N-terminus
        score_n = 0.0
        for c in range(classes):
            prob = self.class_weights_n[c]
            for k in range(self.n_terminus_length):
                # support also unknown characters like 'X'
                if peptide[k] in self.alphabet_index:
                    prob *= self.pssm_n[c, k, self.alphabet_index[peptide[k]]]
            score_n += prob

        # C-terminus
        score_c = 0.0
        c_terminus = peptide[-self.c_terminus_length :]
        for c in range(classes):
            prob = self.class_weights_c[c]
            for k in range(self.c_terminus_length):
                # support also unknown characters like 'X'
                if c_terminus[k] in self.alphabet_index:
                    prob *= self.pssm_c[c, k, self.alphabet_index[c_terminus[k]]]
            score_c += prob

        return score_n * score_c

    def predict(self, peptides: list[str], include_flat: bool = True) -> np.ndarray:
        """Score all peptides in a list.

        Applies function score_peptide() to the peptides.

        Args:
            peptides: The peptides to be scored.
            include_flat: Whether to include the flat motif in the sum of scores.

        Returns:
            The scores.

        Raises:
            NotFittedError: If the model has not yet been fitted.
        """
        if not self.is_fitted:
            raise NotFittedError()

        return np.asarray(
            [self.score_peptide(peptide, include_flat=include_flat) for peptide in peptides]
        )
