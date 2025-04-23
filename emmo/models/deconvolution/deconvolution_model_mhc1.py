"""Module for the MHC1 deconvolution models used in the EM algorithms."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from emmo.constants import MHC1_C_TERMINAL_ANCHORING_LENGTH
from emmo.constants import MHC1_C_TERMINAL_OVERHANG_PENALTY
from emmo.constants import MHC1_LENGTH_RANGE
from emmo.constants import MHC1_N_TERMINAL_ANCHORING_LENGTH
from emmo.constants import MHC1_N_TERMINAL_OVERHANG_PENALTY
from emmo.constants import NATURAL_AAS
from emmo.io.file import load_csv
from emmo.io.file import load_json
from emmo.io.file import Openable
from emmo.io.file import save_csv
from emmo.io.file import save_json
from emmo.io.output import write_matrices
from emmo.models.deconvolution.deconvolution_model_base import DeconvolutionModel


class DeconvolutionModelMHC1(DeconvolutionModel):
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
        self.n_classes = number_of_classes + (1 if has_flat_motif else 0)

        self.has_flat_motif = has_flat_motif

        self.training_params: dict[str, Any] = {}
        self.is_fitted = False

        # position probability matrices
        self.ppm = np.zeros((self.n_classes, self.motif_length, self.n_alphabet))

        # probalitities of the classes and offsets
        self.class_weights: dict[int, np.ndarray] = {}

        # any additional artifacts that should be saved
        self.artifacts = {}

    @property
    def num_of_parameters(self) -> int:
        """Number of model parameters (i.e., frequencies and class/offset weights).

        At the moment, the frequencies from the flat motif are fix and therefore not counted.

        Raises:
            NotFittedError: If the model has not been fitted.

        Returns:
            Number of model parameters.
        """
        # if class_weights is not set, we do not know the number of parameters
        self.check_fitted()

        # class weight incl. flat motif times the no. of offsets
        # (subtract 1 because weights sum to one)
        count = len(self.class_weights) * (self.n_classes - 1)

        # at the moment the flat motif is fix and not estimated
        # count += self.n_alphabet

        # frequencies summed over all other classes
        # (subtract 1 because aa frequencies sum to one)
        count += self.number_of_classes * self.motif_length * (self.n_alphabet - 1)

        return count

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

        model.load_additional_artifacts(directory)
        model.is_fitted = True

        return model

    @classmethod
    def build_from_ppm(
        cls,
        ppm: np.ndarray,
        alphabet: str | tuple[str, ...] | list[str] | None = None,
    ) -> DeconvolutionModelMHC1:
        """Build a model from a single position-probability matrix (PPM).

        Args:
            ppm: The position-probability matrix (PPM) to be used for the model (2-dim. array).
            alphabet: The (amino acid) alphabet. If None, the default alphabet is used.

        Returns:
            The build model consisting of the given PPM.
        """
        if alphabet is None:
            alphabet = NATURAL_AAS

        motif_length = ppm.shape[0]
        if motif_length < MHC1_LENGTH_RANGE[0] or motif_length > MHC1_LENGTH_RANGE[1]:
            raise ValueError(
                f"motif length {motif_length} is not in the range {list(MHC1_LENGTH_RANGE)}"
            )
        if ppm.shape[1] != len(alphabet):
            raise ValueError(
                f"alphabet size {len(alphabet)} does not match PPM size {ppm.shape[1]}"
            )

        model = cls(alphabet, motif_length, 1, has_flat_motif=False)

        model.ppm[0] = ppm

        # the class weights in MHC1 deconvolution models are estimated for each length separately,
        # so we need to set them for all lengths in the length range of MHC1 ligands. As the model
        # consists of a single motif and no flat motif, the class weights vector is [1.0] for each
        # length
        for length in range(MHC1_LENGTH_RANGE[0], MHC1_LENGTH_RANGE[1] + 1):
            model.class_weights[length] = np.asarray([1.0])

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

        self.save_additional_artifacts(directory, force=force)

    def _score_peptide(self, peptide: str, num_classes: int) -> dict[str, Any]:
        """Compute the total score, index of the best class index, and best score.

        Here, the score is the likelihood (no logarithm applied).

        Args:
            peptide: The peptide to be scored.
            num_classes: The number of classes to include (i.e., with or without flat motif).

        Returns:
            A dictionary containing the total score, index of the best class index, best offset,
            and best score.
        """
        length = len(peptide)
        class_weights = self.class_weights[length]

        n_term = self.training_params.get("n_term", MHC1_N_TERMINAL_ANCHORING_LENGTH)
        c_term = self.training_params.get("c_term", MHC1_C_TERMINAL_ANCHORING_LENGTH)
        n_term_penalty = self.training_params.get(
            "n_term_penalty", MHC1_N_TERMINAL_OVERHANG_PENALTY
        )
        c_term_penalty = self.training_params.get(
            "c_term_penalty", MHC1_C_TERMINAL_OVERHANG_PENALTY
        )

        total_score = 0.0
        best_prob = float("-inf")
        best_class = 0
        best_n_overhang = 0
        best_c_overhang = 0

        for c in range(num_classes):
            prob = class_weights[c]
            current_n_overhang = 0
            current_c_overhang = 0

            if length == self.motif_length:
                for pos in range(self.motif_length):
                    prob *= self.ppm[c, pos, self.alphabet_index[peptide[pos]]]

            elif length < self.motif_length:
                for pos in range(n_term):
                    prob *= self.ppm[c, pos, self.alphabet_index[peptide[pos]]]
                for pos in range(length - c_term, length):
                    prob *= self.ppm[
                        c,
                        self.motif_length - length + pos,
                        self.alphabet_index[peptide[pos]],
                    ]

            else:
                max_prob, current_n_overhang, current_c_overhang = self._likelihood_longer_peptide(
                    peptide, self.ppm[c], n_term, c_term, n_term_penalty, c_term_penalty
                )
                prob *= max_prob

            if prob > best_prob:
                best_prob = prob
                best_class = c
                best_n_overhang = current_n_overhang
                best_c_overhang = current_c_overhang

            total_score += prob

        return {
            "score": total_score,
            "best_class": best_class,
            "best_score": best_prob,
            "best_n_overhang": best_n_overhang,
            "best_c_overhang": best_c_overhang,
        }

    def _likelihood_longer_peptide(
        self,
        peptide: str,
        ppm_class: np.ndarray,
        n_term: int,
        c_term: int,
        n_term_penalty: float,
        c_term_penalty: float,
    ) -> tuple[float, int, int]:
        """Compute the likelihood of a peptide longer than the motif length.

        Args:
            peptide: The peptide to be scored.
            ppm_class: The position probability matrix for the class (2-dim. array).
            n_term: The length of the N-terminal part of the sequence to be aligned with the motif
                for sequences that are shorter/longer than 'motif_length'.
            c_term: The length of the C-terminal part of the sequence to be aligned with the motif
                for sequences that are shorter/longer than 'motif_length'.
            n_term_penalty: The penalty factor to be added for N-terminal extensions (sequence
                overhangs the alignment to motif at the N-terminus).
            c_term_penalty: The penalty factor to be added for C-terminal extensions (sequence
                overhangs the alignment to motif at the C-terminus).

        Returns:
            A tuple containing the maximum likelihood, the N-terminal overhang, and the C-terminal
            overhang.
        """
        length = len(peptide)

        current_max = 0.0
        n_overhang = 0
        c_overhang = 0

        for n_start in range(length - self.motif_length + 1):
            for c_start in range(n_start + self.motif_length - c_term, length - c_term + 1):
                prob = (n_term_penalty**n_start) * (c_term_penalty ** (length - c_start - c_term))

                for pos in range(n_term):
                    prob *= ppm_class[
                        pos,
                        self.alphabet_index[peptide[pos + n_start]],
                    ]

                for pos in range(c_term):
                    prob *= ppm_class[
                        self.motif_length - c_term + pos,
                        self.alphabet_index[peptide[pos + c_start]],
                    ]

                if prob > current_max:
                    current_max = prob
                    n_overhang = n_start
                    c_overhang = length - c_start - c_term

        return current_max, n_overhang, c_overhang
