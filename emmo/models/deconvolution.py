"""Module for the deconvolution models used in the EM algorithms."""
from __future__ import annotations

from itertools import product
from typing import Any

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
from emmo.utils.offsets import AlignedOffsets


class DeconvolutionModelMHC1:
    """MHC1 deconvolution model.

    This class holds the model and training parameters, as well as the position probability
    matrices and class weights.
    """

    def __init__(
        self,
        alphabet: str | tuple[str, ...] | list[str],
        motif_length: int,
        number_of_classes: int,
        has_flat_motif: bool = True,
    ) -> None:
        """Initialize the MHC1 deconvolution model.

        Initially, the 'is_fitted' parameter is set to False and needs to be set to True to enable
        prediction.

        Args:
            alphabet: The (amino acid) alphabet.
            motif_length: Motif length.
            number_of_classes: Number of motifs/classes (excl. the flat motif).
            has_flat_motif: Whether the model includes a flat motif.
        """
        self.alphabet = alphabet
        self.n_alphabet = len(alphabet)
        self.alphabet_index = {a: i for i, a in enumerate(alphabet)}

        self.motif_length = motif_length

        self.number_of_classes = number_of_classes
        self.n_classes = number_of_classes + 1

        self.has_flat_motif = has_flat_motif

        self.training_params: dict[str, Any] = {}
        self.is_fitted = False

        self._initialize()

    @classmethod
    def load(cls, directory: Openable) -> DeconvolutionModelMHC1:
        """Load a model from a directory.

        Args:
            directory: The directory containing the model files.

        Returns:
            The loaded model.
        """
        directory = AnyPath(directory)

        model_specs = load_json(directory / "model_specs.json")

        model = cls(
            model_specs["alphabet"],
            model_specs["motif_length"],
            model_specs["number_of_classes"],
        )

        if "training_params" in model_specs:
            model.training_params.update(model_specs["training_params"])

        n = model_specs["number_of_classes"]

        df_class_weights = load_csv(directory / f"class_weights_{n}.csv", index_col=0)

        if len(df_class_weights) != model.n_classes:
            raise ValueError("class weights file is invalid")

        for column in df_class_weights.columns:
            length = int(column.split("_")[1])
            model.class_weights[length] = df_class_weights[column].to_numpy()

        for i in range(model.number_of_classes):
            file = directory / f"matrix_{n}_{i+1}.csv"
            matrix = load_csv(file, index_col=0, header=0).to_numpy()
            if matrix.shape != (model.motif_length, model.n_alphabet):
                raise ValueError(f"invalid frequency matrix file '{file}'")
            model.ppm[i] = matrix

        if model.has_flat_motif:
            file = directory / f"matrix_{n}_flat.csv"
            matrix = load_csv(file, index_col=0, header=0).to_numpy()
            model.ppm[-1] = matrix[0]

        model.is_fitted = True

        return model

    def save(self, directory: Openable, force: bool = False) -> None:
        """Save the model to a directory.

        Args:
            directory: The directory where to save the model.
            force: Overwrite file if they already exist.

        Raises:
            NotFittedError: If the model has not yet been fitted.
        """
        directory = AnyPath(directory)

        if not self.is_fitted:
            raise NotFittedError()

        model_specs = {
            x: self.__dict__[x]
            for x in ["alphabet", "motif_length", "number_of_classes", "training_params"]
        }

        save_json(model_specs, directory / "model_specs.json", force=force)

        write_matrices(
            directory, self.ppm[:-1], self.alphabet, flat_motif=self.ppm[-1, 0, :], force=force
        )

        df_class_weights = pd.DataFrame(
            {f"length_{length}": weights for length, weights in self.class_weights.items()},
            index=list(range(1, self.n_classes)) + ["flat"],
        )
        save_csv(df_class_weights, directory / f"class_weights_{self.n_classes-1}.csv", force=force)

    def _initialize(self) -> None:
        """Init arrays for position probability matrices and dictionary for the class weights."""
        # position probability matrices
        self.ppm = np.zeros((self.n_classes, self.motif_length, self.n_alphabet))

        # probalitities of the classes and offsets
        self.class_weights: dict[int, np.ndarray] = {}

    def get_number_of_parameters(self) -> int:
        """Return the number of parameters (i.e., frequencies and class/offset weights).

        At the moment, the frequencies from the flat motif are fix and therefore not counted.

        Returns:
            Number of parameters.
        """
        # if class_weights is not set, we do not know the number of parameters
        if not self.is_fitted:
            raise NotFittedError()

        # class weight incl. flat motif times the no. of offsets
        # (subtract 1 because weights sum to one)
        count = len(self.class_weights) * (self.n_classes - 1)

        # at the moment the flat motif is fix and not estimated
        # count += self.n_alphabet

        # frequencies summed over all other classes
        # (subtract 1 because aa frequencies sum to one)
        count += self.number_of_classes * self.motif_length * (self.n_alphabet - 1)

        return count


class DeconvolutionModelMHC2:
    """MHC2 deconvolution model.

    This class holds the model and training parameters, as well as the position probability
    matrices and class and offsets weights.
    """

    def __init__(
        self,
        alphabet: str | tuple[str, ...] | list[str],
        motif_length: int,
        number_of_classes: int,
        max_sequence_length: int,
        background: BackgroundType,
        has_flat_motif: bool = True,
    ) -> None:
        """Initialize the MHC2 deconvolution model.

        Initially, the 'is_fitted' parameter is set to False and needs to be set to True to enable
        prediction.

        Args:
            alphabet: The (amino acid) alphabet.
            motif_length: Motif length.
            number_of_classes: Number of motifs/classes (excl. the flat motif).
            max_sequence_length: Maximal sequence length.
            background: The background amino acid frequencies. Can also be a string corresponding
                to one of the available backgrounds.
            has_flat_motif: Whether the model includes a flat motif.
        """
        self.alphabet = alphabet
        self.n_alphabet = len(alphabet)
        self.alphabet_index = {a: i for i, a in enumerate(alphabet)}

        self.motif_length = motif_length

        self.number_of_classes = number_of_classes
        self.n_classes = number_of_classes + (1 if has_flat_motif else 0)

        self.max_sequence_length = max_sequence_length

        self.background = Background(background)

        self.has_flat_motif = has_flat_motif

        self.aligned_offsets = AlignedOffsets(motif_length, max_sequence_length)
        self.n_offsets = self.aligned_offsets.get_number_of_offsets()

        self.training_params: dict[str, Any] = {}
        self.is_fitted = False

        self._initialize_arrays()

    @classmethod
    def load(cls, directory: Openable) -> DeconvolutionModelMHC2:
        """Load a model from a directory.

        Args:
            directory: The directory containing the model files.

        Returns:
            The loaded model.
        """
        directory = AnyPath(directory)

        model_specs = load_json(directory / "model_specs.json")

        model = cls(
            model_specs["alphabet"],
            model_specs["motif_length"],
            model_specs["number_of_classes"],
            model_specs["max_sequence_length"],
            model_specs["background"],
            has_flat_motif=model_specs["has_flat_motif"],
        )

        if "training_params" in model_specs:
            model.training_params.update(model_specs["training_params"])

        n = model_specs["number_of_classes"]

        class_weights = load_csv(directory / f"class_weights_{n}.csv", index_col=0).to_numpy()

        if class_weights.shape != (model.n_classes, model.n_offsets):
            raise ValueError("class weights file is invalid")

        model.class_weights[:] = class_weights

        for i in range(model.number_of_classes):
            file = directory / f"matrix_{n}_{i+1}.csv"
            matrix = load_csv(file, index_col=0, header=0).to_numpy()
            if matrix.shape != (model.motif_length, model.n_alphabet):
                raise ValueError(f"invalid frequency matrix file '{file}'")
            model.ppm[i] = matrix

        if model.has_flat_motif:
            file = directory / f"matrix_{n}_flat.csv"
            matrix = load_csv(file, index_col=0, header=0).to_numpy()
            model.ppm[-1] = matrix[0]

        model.recompute_pssm()
        model.is_fitted = True

        return model

    def save(self, directory: Openable, force: bool = False) -> None:
        """Save the model to a directory.

        Args:
            directory: The directory where to save the model.
            force: Overwrite file if they already exist.

        Raises:
            NotFittedError: If the model has not yet been fitted.
        """
        directory = AnyPath(directory)

        if not self.is_fitted:
            raise NotFittedError()

        model_specs = {
            x: self.__dict__[x]
            for x in [
                "alphabet",
                "motif_length",
                "number_of_classes",
                "max_sequence_length",
                "has_flat_motif",
                "training_params",
            ]
        }
        model_specs["background"] = self.background.get_representation()

        save_json(model_specs, directory / "model_specs.json", force=force)

        write_matrices(
            directory, self.ppm[:-1], self.alphabet, flat_motif=self.ppm[-1, 0, :], force=force
        )

        df_class_weights = pd.DataFrame(
            self.class_weights, index=list(range(1, self.n_classes)) + ["flat"]
        )
        save_csv(df_class_weights, directory / f"class_weights_{self.n_classes-1}.csv", force=force)

    def _initialize_arrays(self) -> None:
        """Initialize arrays.

        Initializes arrays for position probability and scoring matrices, as well as for the class
        and offset weights.
        """
        # position probability matrices
        self.ppm = np.zeros((self.n_classes, self.motif_length, self.n_alphabet))

        # position-specific scoring matrix
        self.pssm = np.zeros_like(self.ppm)

        # probalitities of the classes and offsets
        self.class_weights = np.zeros((self.n_classes, self.n_offsets))

    def recompute_pssm(self) -> None:
        """Update the PSSMs using the current PPMs and the background."""
        self.pssm[:] = self.ppm / self.background.frequencies

    def get_offset_list(self, length: int) -> list[int]:
        """List of valid aligned offsets for a specified length.

        Args:
            length: Sequence length.

        Returns:
            The valid aligned offsets for the sequence length.
        """
        return self.aligned_offsets.get_offset_list(length)

    def get_number_of_parameters(self) -> int:
        """Return the number of parameters (i.e., frequencies and class/offset weights).

        At the moment, the frequencies from the flat motif are fix and therefore not counted.

        Returns:
            Number of parameters.
        """
        # class weight incl. flat motif times the no. of offsets
        # (subtract 1 because weights sum to one)
        count = self.n_offsets * self.n_classes - 1

        # at the moment the flat motif is fix and not estimated
        # count += self.n_alphabet

        # frequencies summed over all other classes (subtract 1 because aa frequencies sum to one)
        count += self.number_of_classes * self.motif_length * (self.n_alphabet - 1)

        return count

    def score_peptide(self, peptide: str, include_flat: bool = False) -> float:
        """Score a peptide with this model.

        The calculates the sum of scores over all classes and all possible offsets. For each class
        and offset, the score is determined by applying the classes' PSSM and multiplying this with
        the class-offset weight.

        Args:
            peptide: The peptide to be scored.
            include_flat: Whether to include the flat motif in the sum of scores.

        Returns:
            The score.
        """
        # TODO: think about how to handle such peptides
        if len(peptide) > self.max_sequence_length:
            return 0.0

        offset_list = self.aligned_offsets.get_offset_list(len(peptide))

        classes = self.n_classes
        if not include_flat and self.has_flat_motif:
            classes -= 1

        score = 0.0

        for c, (i, o) in product(range(classes), enumerate(offset_list)):
            prob = self.class_weights[c, o]
            for k in range(self.motif_length):
                prob *= self.pssm[c, k, self.alphabet_index[peptide[i + k]]]
            score += prob

        return score

    def predict(self, peptides: list[str], include_flat: bool = False) -> np.ndarray:
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


class DeconvolutionModelMHC2NoOffsetWeights:
    """MHC2 deconvolution model without offset weights.

    This class holds the model and training parameters, as well as the position probability
    matrices and class weights.
    """

    def __init__(
        self,
        alphabet: str | tuple[str, ...] | list[str],
        motif_length: int,
        number_of_classes: int,
        background: BackgroundType,
        has_flat_motif: bool = True,
    ) -> None:
        """Initialize the MHC2 deconvolution model.

        Initially, the 'is_fitted' parameter is set to False and needs to be set to True to enable
        prediction.

        Args:
            alphabet: The (amino acid) alphabet.
            motif_length: Motif length.
            number_of_classes: Number of motifs/classes (excl. the flat motif).
            background: The background amino acid frequencies. Can also be a string corresponding
                to one of the available backgrounds.
            has_flat_motif: Whether the model includes a flat motif.
        """
        self.alphabet = alphabet
        self.n_alphabet = len(alphabet)
        self.alphabet_index = {a: i for i, a in enumerate(alphabet)}

        self.motif_length = motif_length

        self.number_of_classes = number_of_classes
        self.n_classes = number_of_classes + (1 if has_flat_motif else 0)

        self.background = Background(background)

        self.has_flat_motif = has_flat_motif

        self.is_fitted = False

        self._initialize_arrays()

    @classmethod
    def load(cls, directory: Openable) -> DeconvolutionModelMHC2NoOffsetWeights:
        """Load a model from a directory.

        Args:
            directory: The directory containing the model files.

        Returns:
            The loaded model.
        """
        directory = AnyPath(directory)

        model_specs = load_json(directory / "model_specs.json")

        model = cls(
            model_specs["alphabet"],
            model_specs["motif_length"],
            model_specs["number_of_classes"],
            model_specs["background"],
            has_flat_motif=model_specs["has_flat_motif"],
        )

        n = model_specs["number_of_classes"]

        class_weights = load_csv(directory / f"class_weights_{n}.csv", index_col=0).to_numpy()

        if class_weights.shape != (model.n_classes, 1):
            raise ValueError(
                "class weights file is invalid: required shape "
                f"{(model.n_classes, )} but got {class_weights.shape}"
            )

        model.class_weights[:] = class_weights[:, 0]

        for i in range(model.number_of_classes):
            file = directory / f"matrix_{n}_{i+1}.csv"
            matrix = load_csv(file, index_col=0, header=0).to_numpy()
            if matrix.shape != (model.motif_length, model.n_alphabet):
                raise ValueError(f"invalid frequency matrix file '{file}'")
            model.ppm[i] = matrix

        if model.has_flat_motif:
            file = directory / f"matrix_{n}_flat.csv"
            matrix = load_csv(file, index_col=0, header=0).to_numpy()
            model.ppm[-1] = matrix[0]

        model.recompute_pssm()
        model.is_fitted = True

        return model

    def save(self, directory: Openable, force: bool = False) -> None:
        """Save the model to a directory.

        Args:
            directory: The directory where to save the model.
            force: Overwrite file if they already exist.

        Raises:
            NotFittedError: If the model has not yet been fitted.
        """
        directory = AnyPath(directory)

        if not self.is_fitted:
            raise NotFittedError()

        model_specs = {
            x: self.__dict__[x]
            for x in [
                "alphabet",
                "motif_length",
                "number_of_classes",
                "has_flat_motif",
            ]
        }
        model_specs["background"] = self.background.get_representation()

        save_json(model_specs, directory / "model_specs.json", force=force)

        write_matrices(
            directory, self.ppm[:-1], self.alphabet, flat_motif=self.ppm[-1, 0, :], force=force
        )

        # write the class weights
        df_class_weights = pd.DataFrame(
            self.class_weights, index=list(range(1, self.n_classes)) + ["flat"]
        )
        save_csv(df_class_weights, directory / f"class_weights_{self.n_classes-1}.csv", force=force)

    def _initialize_arrays(self) -> None:
        """Initialize arrays.

        Initializes arrays for the position probability and scoring matrices, as well as for the
        class weights.
        """
        # position probability matrices
        self.ppm = np.zeros((self.n_classes, self.motif_length, self.n_alphabet))

        # position-specific scoring matrix
        self.pssm = np.zeros_like(self.ppm)

        # probalitities of the classes and offsets
        self.class_weights = np.zeros(self.n_classes)

    def recompute_pssm(self) -> None:
        """Update the PSSMs using the current PPMs and the background."""
        self.pssm[:] = self.ppm / self.background.frequencies

    def get_number_of_parameters(self) -> int:
        """Return the number of parameters (frequencies and class priors).

        Returns
            The number of parameters.
        """
        # class weight incl. flat motif (subtract 1 because weights sum to one)
        count = self.n_classes - 1

        # at the moment the flat motif is fix and not estimated
        # count += self.n_alphabet

        # frequencies summed over all other classes (subtract 1 because aa frequencies sum to one)
        count += self.number_of_classes * self.motif_length * (self.n_alphabet - 1)

        return count

    def score_peptide(self, peptide: str, include_flat: bool = False) -> float:
        """Score a peptide with this model.

        The calculates the sum of scores over all classes and all possible offsets. For each class
        and offset, the score is determined by applying the classes' PSSM and multiplying this with
        the class weight.

        Args:
            peptide: The peptide to be scored.
            include_flat: Whether to include the flat motif in the sum of scores.

        Returns:
            The score.
        """
        # TODO: think about how to handle such peptides
        if len(peptide) < self.motif_length:
            return 0.0

        classes = self.n_classes
        if not include_flat and self.has_flat_motif:
            classes -= 1

        score = 0.0

        for c in range(classes):
            current_max = float("-inf")
            for i in range(0, len(peptide) - self.motif_length + 1):
                prob = self.class_weights[c]
                for k in range(self.motif_length):
                    prob *= self.pssm[c, k, self.alphabet_index[peptide[i + k]]]
                if prob > current_max:
                    current_max = prob
            score += current_max

        return score

    def predict(self, peptides: list[str], include_flat: bool = False) -> np.ndarray:
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


if __name__ == "__main__":
    from emmo.constants import REPO_DIRECTORY

    input_name = "HLA-A0101_A0218_background_class_II"
    directory = REPO_DIRECTORY / "validation" / "local" / input_name

    model = DeconvolutionModelMHC2.load(directory)

    df = load_csv(
        REPO_DIRECTORY / "validation" / "local" / f"{input_name}.txt",
        header=None,
        names=["peptide"],
    )
    df["score"] = model.predict(df["peptide"])

    print(df)
    save_csv(df, REPO_DIRECTORY / "validation" / "local" / f"{input_name}_scored.csv", force=True)
