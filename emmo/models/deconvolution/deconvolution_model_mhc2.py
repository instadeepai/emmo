"""Module for the MHC2 deconvolution models used in the EM algorithms."""
from __future__ import annotations

from abc import abstractmethod
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
from emmo.models.deconvolution.deconvolution_model_base import DeconvolutionModel
from emmo.pipeline.background import Background
from emmo.pipeline.background import BackgroundType
from emmo.utils.offsets import AlignedOffsets


class BaseDeconvolutionModelMHC2(DeconvolutionModel):
    """Base class for MHC2 deconvolution models."""

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

        self.training_params: dict[str, Any] = {}
        self.is_fitted = False

        # position probability matrices
        self.ppm = np.zeros((self.n_classes, self.motif_length, self.n_alphabet))

        # position-specific scoring matrix
        self.pssm = np.zeros_like(self.ppm)

        self.class_weights = self._initialize_class_weights()

        # any additional artifacts that should be saved
        self.artifacts = {}

    @property
    @abstractmethod
    def num_of_parameters(self) -> int:
        """Number of model parameters.

        Returns:
            Number of model parameters.
        """

    @classmethod
    @abstractmethod
    def load(cls, directory: Openable) -> DeconvolutionModel:
        """Load a model from a directory.

        Args:
            directory: The directory containing the model files.

        Returns:
            The loaded model.
        """

    @abstractmethod
    def save(self, directory: Openable, force: bool = False) -> None:
        """Save the model to a directory.

        Args:
            directory: The directory where to save the model.
            force: Overwrite file if they already exist.
        """

    def recompute_pssm(self) -> None:
        """Update the PSSMs using the current PPMs and the background."""
        self.pssm[:] = self.ppm / self.background.frequencies

    @abstractmethod
    def _initialize_class_weights(self) -> np.ndarray:
        """Initialize class weights.

        Returns:
            The class weights array initialized with zeros.
        """


class DeconvolutionModelMHC2(BaseDeconvolutionModelMHC2):
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
        self.aligned_offsets = AlignedOffsets(motif_length, max_sequence_length)
        self.n_offsets = self.aligned_offsets.get_number_of_offsets()
        self.max_sequence_length = max_sequence_length
        super().__init__(
            alphabet, motif_length, number_of_classes, background, has_flat_motif=has_flat_motif
        )

    @property
    def num_of_parameters(self) -> int:
        """Number of model parameters (i.e., frequencies and class/offset weights).

        At the moment, the frequencies from the flat motif are fix and therefore not counted.

        Returns:
            Number of model parameters.
        """
        # class weight incl. flat motif times the no. of offsets
        # (subtract 1 because weights sum to one)
        count = self.n_offsets * self.n_classes - 1

        # at the moment the flat motif is fix and not estimated
        # count += self.n_alphabet

        # frequencies summed over all other classes (subtract 1 because aa frequencies sum to one)
        count += self.number_of_classes * self.motif_length * (self.n_alphabet - 1)

        return count

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
        model.load_additional_artifacts(directory)
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

        self.check_fitted()

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

        self.save_additional_artifacts(directory, force=force)

    def get_offset_list(self, length: int) -> list[int]:
        """List of valid aligned offsets for a specified length.

        Args:
            length: Sequence length.

        Returns:
            The valid aligned offsets for the sequence length.
        """
        return self.aligned_offsets.get_offset_list(length)

    def best_offset_per_class(self) -> dict[str, int]:
        """Return the highest-weight offset per class of the model.

        The class label strings are 1-based ('1', '2', ...) and 'flat' for the flat motif. The
        offsets are the aligned offsets (0 ... middle offsets, < 0 ... shifted towards N-terminus,
        > 0 ... shifted towards C-terminus).

        Returns:
            A dictionary with the best offset per class.
        """
        self.check_fitted()

        offset_annotations = self.aligned_offsets.get_offset_annotation()
        class_annotations = [str(i + 1) for i in range(self.number_of_classes)]
        if self.has_flat_motif:
            class_annotations.append("flat")

        class2best_offset = {}
        for i, c in enumerate(class_annotations):
            idx_max = self.class_weights[i].argmax()
            class2best_offset[c] = offset_annotations[idx_max]

        return class2best_offset

    def _initialize_class_weights(self) -> np.ndarray:
        """Initialize class weights.

        Returns:
            The class weights array initialized with zeros.
        """
        # probalitities of the classes and offsets
        return np.zeros((self.n_classes, self.n_offsets))

    def _score_peptide(self, peptide: str, num_classes: int) -> dict[str, Any]:
        """Compute the total score, index of the best class index, best offset, and best score.

        Args:
            peptide: The peptide to be scored.
            num_classes: The number of classes to include (i.e., with or without flat motif).

        Returns:
            A dictionary containing the total score, index of the best class index, best offset,
            and best score.
        """
        # TODO: think about how to handle such peptides
        if len(peptide) > self.max_sequence_length:
            raise ValueError(
                f"peptide {peptide} is longer than maximal sequence length of "
                f"{self.max_sequence_length}"
            )

        offset_list = self.aligned_offsets.get_offset_list(len(peptide))

        total_score = 0.0

        best_prob = float("-inf")
        best_class = 0
        best_offset = 0

        for c, (i, o) in product(range(num_classes), enumerate(offset_list)):
            prob = self.class_weights[c, o]
            for k in range(self.motif_length):
                prob *= self.pssm[c, k, self.alphabet_index[peptide[i + k]]]

            if prob > best_prob:
                best_prob = prob
                best_class = c
                best_offset = o

            total_score += prob

        return {
            "score": total_score,
            "best_class": best_class,
            "best_offset": self.aligned_offsets.get_offset_in_sequence(len(peptide), best_offset),
            "best_score": best_prob,
        }


class DeconvolutionModelMHC2NoOffsetWeights(BaseDeconvolutionModelMHC2):
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
        super().__init__(
            alphabet, motif_length, number_of_classes, background, has_flat_motif=has_flat_motif
        )

    @property
    def num_of_parameters(self) -> int:
        """Number of model parameters (frequencies and class priors).

        Returns
            The number of model parameters.
        """
        # class weight incl. flat motif (subtract 1 because weights sum to one)
        count = self.n_classes - 1

        # at the moment the flat motif is fix and not estimated
        # count += self.n_alphabet

        # frequencies summed over all other classes (subtract 1 because aa frequencies sum to one)
        count += self.number_of_classes * self.motif_length * (self.n_alphabet - 1)

        return count

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
        model.load_additional_artifacts(directory)
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

        self.check_fitted()

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

        self.save_additional_artifacts(directory, force=force)

    def _initialize_class_weights(self) -> np.ndarray:
        """Initialize class weights.

        Returns:
            The class weights array initialized with zeros.
        """
        return np.zeros(self.n_classes)

    def _score_peptide(self, peptide: str, num_classes: int) -> dict[str, Any]:
        """Compute the total score, index of the best class index best offset, and best score.

        Args:
            peptide: The peptide to be scored.
            classes: The number of classes to include (i.e., with or without flat motif).

        Returns:
            A dictionary containing the total score, index of the best class index, best offset,
            and best score.
        """
        total_score = 0.0

        best_prob = float("-inf")
        best_class = 0
        best_offset = 0

        for c in range(num_classes):
            current_max = float("-inf")
            for i in range(0, len(peptide) - self.motif_length + 1):
                prob = self.class_weights[c]

                for k in range(self.motif_length):
                    prob *= self.pssm[c, k, self.alphabet_index[peptide[i + k]]]

                if prob > current_max:
                    current_max = prob

                if prob > best_prob:
                    best_prob = prob
                    best_class = c
                    best_offset = i

            total_score += current_max

        return {
            "score": total_score,
            "best_class": best_class,
            "best_offset": best_offset,
            "best_score": best_prob,
        }
