"""Test cases for the pipeline background functions."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pytest

from emmo.constants import NATURAL_AAS
from emmo.io.file import save_json
from emmo.pipeline.background import Background
from emmo.pipeline.background import BackgroundType


@pytest.fixture(scope="module")
def example_background() -> Background:
    """Return an example instance of Background."""
    return Background("uniprot")


@pytest.fixture(scope="module")
def expected_named_background_frequencies() -> dict[str, list[float]]:
    """Expected frequencies for backgrounds that can be accessed by their names."""
    return {
        "uniprot": [
            0.0702,
            0.0230,
            0.0473,
            0.0710,
            0.0365,
            0.0657,
            0.0263,
            0.0433,
            0.0572,
            0.0996,
            0.0213,
            0.0359,
            0.0631,
            0.0477,
            0.0564,
            0.0833,
            0.0536,
            0.0597,
            0.0122,
            0.0267,
        ],
        "psi_blast": [
            0.078047,
            0.019246,
            0.053640,
            0.062949,
            0.038556,
            0.073772,
            0.021992,
            0.051420,
            0.057438,
            0.090191,
            0.022425,
            0.044873,
            0.052028,
            0.042644,
            0.051295,
            0.071198,
            0.058413,
            0.064409,
            0.013298,
            0.032165,
        ],
        "MHC1_biondeep": [
            0.076584,
            0.005569,
            0.044324,
            0.063805,
            0.055308,
            0.043486,
            0.030039,
            0.060332,
            0.049878,
            0.114874,
            0.018129,
            0.029773,
            0.061633,
            0.038834,
            0.051136,
            0.067713,
            0.053714,
            0.080919,
            0.010045,
            0.043906,
        ],
        "MHC2_biondeep": [
            0.086865,
            0.004767,
            0.049174,
            0.070215,
            0.031256,
            0.067076,
            0.025248,
            0.051752,
            0.066980,
            0.086119,
            0.012101,
            0.038058,
            0.072285,
            0.048338,
            0.058607,
            0.076822,
            0.052485,
            0.070511,
            0.006022,
            0.025322,
        ],
    }


@pytest.mark.parametrize(
    "background",
    [
        "uniprot",
        "psi_blast",
        np.ones((len(NATURAL_AAS),), dtype=np.float64) / len(NATURAL_AAS),
        np.arange(len(NATURAL_AAS)) / np.arange(len(NATURAL_AAS)).sum(),
    ],
)
def test_background_init(background: BackgroundType) -> None:
    """Test that initialization of 'Background' works as expected.

    Args:
        background: The background frequencies or the name of an available background distribution.
    """
    background_obj = Background(background)
    assert background_obj.frequencies.shape == (len(NATURAL_AAS),)
    assert np.isclose(np.sum(background_obj.frequencies), 1.0)

    if isinstance(background, str):
        assert background_obj.name == background
    else:
        assert background_obj.name is None


def test_background_init_wrong_shape() -> None:
    """Test that initialization of 'Background' raises an error if the input has the wrong shape."""
    with pytest.raises(ValueError, match="background frequencies must have shape"):
        Background(
            np.ones(((len(NATURAL_AAS) - 1),), dtype=np.float64) / (len(NATURAL_AAS) - 1),
        )


def test_background_init_wrong_sum() -> None:
    """Test that initialization of 'Background' raises an error if the sum is not 1."""
    with pytest.raises(ValueError, match="background frequencies must sum up to 1"):
        Background(
            np.ones((len(NATURAL_AAS),), dtype=np.float64) / (len(NATURAL_AAS) - 1),
        )


@pytest.mark.parametrize(
    "background",
    ["uniprot", "psi_blast", "MHC1_biondeep", "MHC2_biondeep"],
)
def test_get_background_by_name(
    background: str,
    expected_named_background_frequencies: dict[str, list[float]],
) -> None:
    """Test that getting the background frequencies by name returns the expected frequencies.

    Args:
        background: Name of the background distribution.
        expected_named_background_frequencies: The expected frequencies.
    """
    background_obj = Background(background)
    assert np.allclose(
        background_obj.frequencies, expected_named_background_frequencies[background]
    )


def test_from_sequences(
    example_sequences: list[str],
    expected_amino_acid_frequencies: list[float],
) -> None:
    """Test that 'from_sequences' works as expected."""
    assert np.allclose(
        Background.from_sequences(example_sequences).frequencies,
        expected_amino_acid_frequencies,
    )


@pytest.mark.parametrize(
    ("background", "expected_representation"),
    [
        ("uniprot", "uniprot"),
        ("psi_blast", "psi_blast"),
        (
            np.ones((len(NATURAL_AAS),), dtype=np.float64) / len(NATURAL_AAS),
            len(NATURAL_AAS) * [1 / len(NATURAL_AAS)],
        ),
    ],
)
def test_get_representation(
    background: BackgroundType,
    expected_representation: str | list[float],
) -> None:
    """Test that 'get_representation' works as expected."""
    assert Background(background).get_representation() == expected_representation


@pytest.mark.parametrize(
    ("name", "distribution_name"), [("test1", "uniprot"), ("test2", "psi_blast"), (None, "uniprot")]
)
def test_load(
    tmp_directory: Path,
    expected_named_background_frequencies: dict[str, list[float]],
    name: str | None,
    distribution_name: str,
) -> None:
    """Test that loading a background distribution works as expected.

    Args:
        tmp_directory: Directory where to save the file.
        expected_named_background_frequencies: The expected frequencies dictionary.
        name: Name of the background distribution.
        distribution_name: Name of the distribution from which to take the frequencies (Key in
            'expected_named_background_frequencies')
    """
    file_path = tmp_directory / "test_background.json"
    data = {
        "name": name,
        "frequencies": dict(
            zip(NATURAL_AAS, expected_named_background_frequencies[distribution_name])
        ),
    }
    save_json(data, file_path, force=True)

    background = Background.load(file_path)

    assert background.name == name
    assert np.allclose(
        background.frequencies, expected_named_background_frequencies[distribution_name]
    )


def test_load_with_wrong_keys(
    tmp_directory: Path,
) -> None:
    """Test that loading a background distribution raises an error if name has the wrong type.

    Args:
        tmp_directory: Directory where to save the file.
    """
    file_path = tmp_directory / "test_background.json"
    data = {
        "wrong_key1": 1,
        "frequencies": dict(
            zip(NATURAL_AAS, np.ones((len(NATURAL_AAS),), dtype=np.float64) / len(NATURAL_AAS))
        ),
    }
    save_json(data, file_path, force=True)

    with pytest.raises(ValueError, match="The JSON file needs to contain the keys:"):
        Background.load(file_path)


def test_load_with_wrong_type_of_name(
    tmp_directory: Path,
    expected_named_background_frequencies: dict[str, list[float]],
) -> None:
    """Test that loading a background distribution raises an error if name has the wrong type.

    Args:
        tmp_directory: Directory where to save the file.
        expected_named_background_frequencies: The expected frequencies dictionary.
    """
    file_path = tmp_directory / "test_background.json"
    data = {
        # wrong type int
        "name": 1,
        "frequencies": dict(zip(NATURAL_AAS, expected_named_background_frequencies["uniprot"])),
    }
    save_json(data, file_path, force=True)

    with pytest.raises(
        TypeError, match="Name of the background distribution must be a string or None"
    ):
        Background.load(file_path)


@pytest.mark.parametrize("force", [True, False])
@patch("emmo.pipeline.background.save_json")
def test_save(
    mock_save_json: MagicMock,
    force: bool,
    tmp_directory: Path,
    example_background: Background,
) -> None:
    """Ensure save_json is called."""
    file_path = tmp_directory / "example_background.json"
    example_background.save(file_path, force=force)
    data = {
        "name": example_background.name,
        "frequencies": dict(zip(NATURAL_AAS, example_background.frequencies)),
    }

    mock_save_json.assert_called_with(data, file_path, force=force)


def test_save_already_exist_no_force(tmp_directory: Path, example_background: Background) -> None:
    """Ensure an error is raised if file_path already exists and force=False.

    Args:
        tmp_directory: Directory where to save the file.
        example_background_object: Example instance of Background.
    """
    file_path = tmp_directory / "test_background_save_already_exist_no_force.json"

    file_path.touch()

    with pytest.raises(FileExistsError):
        example_background.save(file_path, force=False)
