"""Module used to define helper functions used in several test files."""
from __future__ import annotations

from pathlib import Path

from cloudpathlib import CloudPath


def get_common_model_artifact_file_paths(model_path: CloudPath | Path) -> list[CloudPath | Path]:
    """Get some example model artifacts file paths for testing purposes.

    It will return the paths:
    {model_path}/
        model_specs.json
        testing/
            metrics1.csv
            metrics2.csv
    """
    return [
        model_path / "model_specs.json",
        model_path / "testing" / "metrics1.csv",
        model_path / "testing" / "metrics2.csv",
    ]


def create_common_model_artifacts(model_path: CloudPath | Path) -> None:
    """Create some example model artifacts for testing purposes.

    It will create:
    {model_path}/
        model_specs.json
        testing/
            metrics1.csv
            metrics2.csv
    """
    for file_path in get_common_model_artifact_file_paths(model_path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.touch()
