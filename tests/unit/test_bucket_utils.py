"""Test cases for the utils bucket functions."""
from __future__ import annotations

from pathlib import Path

import pytest
from cloudpathlib import CloudPath

from emmo.bucket.utils import get_bucket_path
from emmo.bucket.utils import get_local_path
from emmo.constants import DATA_DIRECTORY
from emmo.constants import MODELS_DIRECTORY


@pytest.mark.parametrize(
    ("bucket_path", "expected_local_path"),
    [
        ("gs://biondeep-data/test_file.csv", DATA_DIRECTORY / "test_file.csv"),
        (
            "gs://biondeep-data/directory/test_file.csv",
            DATA_DIRECTORY / "directory" / "test_file.csv",
        ),
        (
            "gs://biondeep-models/emmo/binding_predictor/model_name",
            MODELS_DIRECTORY / "binding_predictor" / "model_name",
        ),
        (
            "s3://biondeep-data/test_file.csv",
            DATA_DIRECTORY / "test_file.csv",
        ),
    ],
)
def test_get_local_path(bucket_path: str, expected_local_path: Path) -> None:
    """Ensure get_local_path is working as expected."""
    assert str(get_local_path(bucket_path)) == str(expected_local_path)


def test_get_local_path_unknown_prefix() -> None:
    """Ensure an error is raised when an unknown prefix is provided."""
    msg = "does not start with a known bucket prefix"
    with pytest.raises(ValueError, match=msg):
        get_local_path("unknown://biondeep-data/test_file.csv")


def test_get_local_path_unknown_bucket() -> None:
    """Ensure an error is raised when an unknown bucket is provided."""
    msg = "The bucket used is unknown. Currently known buckets are"
    with pytest.raises(ValueError, match=msg):
        get_local_path("s3://unknown-bucket/test_file.csv")


@pytest.mark.parametrize(
    ("local_path", "expected_bucket_path"),
    [
        (DATA_DIRECTORY / "test_file.csv", "gs://biondeep-data/test_file.csv"),
        (
            DATA_DIRECTORY / "directory" / "test_file.csv",
            "gs://biondeep-data/directory/test_file.csv",
        ),
        (
            MODELS_DIRECTORY / "binding_predictor" / "model_name",
            "gs://biondeep-models/emmo/binding_predictor/model_name",
        ),
        (
            MODELS_DIRECTORY / "cleavage" / "model_name2",
            "gs://biondeep-models/emmo/cleavage/model_name2",
        ),
    ],
)
def test_get_bucket_path(local_path: Path, expected_bucket_path: str) -> None:
    """Ensure get_local_path is working as expected."""
    assert get_bucket_path(local_path) == CloudPath(expected_bucket_path)


def test_get_bucket_path_unsupported_local_path() -> None:
    """Ensure an error is raised when the local path is in the data or models folder."""
    msg = "Only paths to files/directories in the local"
    with pytest.raises(ValueError, match=msg):
        get_bucket_path(Path("/tmp/random_folder/test_file.csv"))
