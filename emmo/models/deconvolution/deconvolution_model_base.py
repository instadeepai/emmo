"""Module for the base deconvolution model."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Any

import numpy as np
import pandas as pd
from cloudpathlib import AnyPath

from emmo.io.file import load_csv
from emmo.io.file import load_json
from emmo.io.file import load_npy
from emmo.io.file import Openable
from emmo.io.file import save_csv
from emmo.io.file import save_json
from emmo.io.file import save_npy
from emmo.utils.exceptions import NotFittedError


class DeconvolutionModel(ABC):
    """Abstract base class for deconvolution models."""

    number_of_classes: int
    n_classes: int
    has_flat_motif: bool
    alphabet: str | tuple[str, ...] | list[str]
    ppm: np.ndarray
    artifacts: dict[str, Any]
    is_fitted: bool

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

    def check_fitted(self) -> None:
        """Check if the model is fitted.

        Raises:
            NotFittedError: If the model has not yet been fitted.
        """
        if not self.is_fitted:
            raise NotFittedError()

    def save_additional_artifacts(self, directory: Openable, force: bool = False) -> None:
        """Save additional artifacts to a subdirectory 'artifacts' in the given directory.

        Args:
            directory: The directory where to save the artifacts in a subdirectory 'artifacts'.
            force: Overwrite files if they already exist.
        """
        artifacts_directory = AnyPath(directory) / "artifacts"

        additional_artifacts: dict[str, Any] = {}
        for name, artifact in self.artifacts.items():
            if isinstance(artifact, pd.DataFrame):
                save_csv(artifact, artifacts_directory / f"{name}.csv", force=force)
            elif isinstance(artifact, np.ndarray):
                save_npy(artifact, artifacts_directory / f"{name}.npy", force=force)
            else:
                additional_artifacts[name] = artifact

        if additional_artifacts:
            save_json(
                additional_artifacts,
                artifacts_directory / "additional_artifacts.json",
                force=force,
            )

    def load_additional_artifacts(self, directory: Openable) -> None:
        """Load additional artifacts from a subdirectory 'artifacts' in the given directory.

        Args:
            directory: The model directory where to load the artifacts from a subdirectory
                'artifacts'.
        """
        artifacts_directory = AnyPath(directory) / "artifacts"

        if not artifacts_directory.is_dir():
            return

        for path in artifacts_directory.iterdir():
            if not path.is_file():
                continue

            if path.suffix == ".csv":
                artifact = load_csv(path)
            elif path.suffix == ".npy":
                artifact = load_npy(path)
            elif path.name == "additional_artifacts.json":
                additional_artifacts = load_json(path)
                self.artifacts.update(additional_artifacts)
                continue
            else:
                raise ValueError(f"unsupported artifact file '{path.name}'")

            self.artifacts[path.stem] = artifact

    def score_peptide(self, peptide: str, include_flat: bool = True) -> dict[str, Any]:
        """Score a peptide with this model.

        The function calculates the total score/likelihood over all classes (and all possible
        offsets), the best class, the best score corresponding to the best class (and offset), and
        additional information depending on the model type (e.g., the best offset).

        Args:
            peptide: The peptide to be scored.
            include_flat: Whether to include the flat motif in the sum of scores. This has no effect
                if the model does not have a flat motif.

        Returns:
            A dictionary with the total score, the best class (1-based / 'flat'), the best score
            are other information.
        """
        num_classes = self.n_classes
        if not include_flat and self.has_flat_motif:
            num_classes -= 1

        result = self._score_peptide(peptide, num_classes)

        if include_flat and self.has_flat_motif and result["best_class"] == self.n_classes - 1:
            result["best_class"] = "flat"
        else:
            result["best_class"] = str(result["best_class"] + 1)

        return result

    def predict(self, peptides: list[str] | pd.Series, include_flat: bool = True) -> pd.DataFrame:
        """Score all peptides in a list.

        Applies function score_peptide() to the peptides.

        Args:
            peptides: The peptides to be scored.
            include_flat: Whether to include the flat motif in the sum of scores.

        Returns:
            A DataFrame containing the following columns and additional columns depending on the
            model:
            - 'score': The total score per peptide summed over all classes (and offsets).
            - 'best_class': The best class (1-based / 'flat') per peptide.
            - 'best_score': The best score corresponding to best class (and offset) per peptide.

        Raises:
            NotFittedError: If the model has not yet been fitted.
        """
        self.check_fitted()

        return pd.DataFrame(
            [self.score_peptide(peptide, include_flat=include_flat) for peptide in peptides]
        )

    def _score_peptide(self, peptide: str, num_classes: int) -> dict[str, Any]:
        """Score a peptide with this model.

        Child classes should implement this method.

        Args:
            peptide: The peptide to be scored.
            num_classes: The number of classes to include (i.e., with or without flat motif).

        Returns:
            A dictionary containing the score and additional information.
        """
        raise NotImplementedError()
