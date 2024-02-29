"""Module for handling background frequencies."""
from __future__ import annotations

from typing import Iterable
from typing import Union

import numpy as np

from emmo.constants import NATURAL_AAS
from emmo.io.file import load_json
from emmo.io.file import Openable
from emmo.io.file import save_json
from emmo.resources.background_frequencies import get_background
from emmo.utils.motifs import total_frequencies


BackgroundType = Union[str, list[float], tuple[float, ...], np.ndarray, "Background"]


class Background:
    """Class that handles background frequencies."""

    def __init__(self, background: BackgroundType) -> None:
        """Initialize the Background class.

        Args:
            background: The background to be used. Can be another Background instance, a list/tuple
                of floats, a numpy.ndarray, or a string corresponding to one of the available
                background frequencies (e.g., 'uniprot' or 'psi_blast').

        Raises:
            ValueError: If a string was provided that does not belong to any of the available
                background frequencies.
            ValueError: If the provided frequencies do not have the correct shape.
            ValueError: If the provided frequencies do not sum up to 1.
        """
        self._name: str | None = None

        if isinstance(background, Background):
            self._frequencies = background.frequencies
            self._name = background.name
        elif isinstance(background, (list, tuple, np.ndarray)):
            self._frequencies = np.array(background, dtype=np.float64)
        elif isinstance(background, str):
            self._frequencies = Background._get_background_by_name(background)
            self._name = background
        else:
            raise TypeError(f"type {type(background)} is not supported for initialization")

        self._frequencies.flags.writeable = False

        # TODO: generalize the Background class for arbitrary alphabets if needed in the future
        if self._frequencies.shape != (len(NATURAL_AAS),):
            raise ValueError(
                f"background frequencies must have shape {(len(NATURAL_AAS),)} but got "
                f"{self._frequencies.shape}"
            )

        if not np.isclose(np.sum(self._frequencies), 1.0):
            raise ValueError("background frequencies must sum up to 1")

    @property
    def name(self) -> str | None:
        """Getter for the name."""
        return self._name

    @property
    def frequencies(self) -> np.ndarray:
        """Getter for the frequencies."""
        return self._frequencies

    @staticmethod
    def _get_background_by_name(name: str) -> np.ndarray:
        """Get the background distribution from the set of available ones.

        Args:
            name: The name of the background distribution.

        Raises:
            ValueError: If the specified name is not available.

        Returns:
            The background distribution.
        """
        return get_background(name)

    @classmethod
    def load(cls, file_path: Openable) -> Background:
        """Load a serialized background distribution.

        Args:
            file_path: Path to the JSON file with the background distribution.
        """
        background_dict = load_json(file_path)

        needed_keys = {"name", "frequencies"}
        if not needed_keys.issubset(background_dict):
            raise ValueError(f"The JSON file needs to contain the keys: {list(needed_keys)}")

        name = background_dict["name"]
        if name is not None and not isinstance(name, str):
            raise TypeError("Name of the background distribution must be a string or None")

        amino_acid2frequency = background_dict["frequencies"]
        frequencies = np.asarray([amino_acid2frequency[aa] for aa in NATURAL_AAS], dtype=np.float64)

        background = Background(frequencies)
        background._name = name

        return background

    @classmethod
    def from_sequences(cls, sequences: Iterable[str]) -> Background:
        """Return frequencies computed from a list of sequences.

        Args:
            sequences: The sequences.
        """
        return Background(total_frequencies(sequences))

    def save(self, file_path: Openable, force: bool = False) -> None:
        """Save the background distribution to a JSON file.

        Args:
            file_path: File path where to save the background distribution.
            force: Overwrite existing file.
        """
        data = {
            "name": self.name,
            "frequencies": dict(zip(NATURAL_AAS, self.frequencies)),
        }

        save_json(data, file_path, force=force)

    def get_representation(self) -> str | list[float]:
        """Get a representation e.g. for writing the background to a JSON file.

        If the frequencies are available in the package, the associated name is returned.
        Otherwise, the frequencies are directly returned as a list.
        """
        if self.name:
            return self.name
        else:
            return self.frequencies.tolist()
