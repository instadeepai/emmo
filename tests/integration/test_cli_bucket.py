"""Test cases for the functions defined in biondeep/cli/bucket."""
from __future__ import annotations

from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from click.testing import CliRunner
from cloudpathlib import CloudPath

from emmo.cli.bucket import pull_model
from emmo.cli.bucket import push_model
from emmo.constants import MODELS_BUCKET_NAME
from emmo.constants import MODELS_DIRECTORY


@pytest.mark.parametrize("force", [True, False])
@patch("emmo.cli.bucket.get_model_puller")
def test_pull_model(
    mock_get_model_puller: MagicMock,
    force: bool,
) -> None:
    """Ensure the model puller's functions get called with the proper parameters in pull-model."""
    model_puller = mock_get_model_puller.return_value
    model_puller.should_pull.return_value = True

    model_uri = f"gs://{MODELS_BUCKET_NAME}/dummy_model"
    model_cloud_path = CloudPath(model_uri)
    model_cloud_path.touch()

    parameters = [
        "--model_name",
        model_uri,
    ]

    if force:
        parameters.append("--force")

    runner = CliRunner()
    result = runner.invoke(pull_model, parameters, catch_exceptions=False)
    assert result.exit_code == 0

    model_puller.should_pull.assert_called_with(model_cloud_path, force)
    model_puller.pull.assert_called_with(model_cloud_path)


@pytest.mark.parametrize("force", [True, False])
@patch("emmo.cli.bucket.get_model_path_from_name")
@patch("emmo.cli.bucket.get_model_pusher")
def test_push_model(
    mock_get_model_pusher: MagicMock,
    mock_get_model_path_from_name: MagicMock,
    force: bool,
) -> None:
    """Ensure the model pusher's functions get called with the proper parameters in push-model."""
    model_name = "dummy_model"
    model_path = MODELS_DIRECTORY / "binding_predictor" / model_name
    model_pusher = mock_get_model_pusher.return_value
    model_pusher.should_push.return_value = True
    mock_get_model_path_from_name.return_value = model_path

    parameters = [
        "--model_name",
        model_name,
    ]

    if force:
        parameters.append("--force")

    runner = CliRunner()
    result = runner.invoke(push_model, parameters, catch_exceptions=False)
    assert result.exit_code == 0

    mock_get_model_pusher.assert_called_with(model_path)
    model_pusher.should_push.assert_called_with(model_path, force)
    model_pusher.push.assert_called_with(model_path)
