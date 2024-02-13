"""Test cases for the classes defined in emmo/bucket/model_puller.py."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from cloudpathlib import CloudPath

from emmo.bucket.model_puller import BaseModelPuller
from emmo.bucket.model_puller import DefaultModelPuller
from emmo.bucket.model_puller import get_model_puller
from tests.helpers import create_common_model_artifacts
from tests.helpers import get_common_model_artifact_file_paths


@contextlib.contextmanager
def pull_ctx(models_directory: Path) -> Iterator[None]:
    """Context manager to use during pulling to pull the models in a tmp directory."""
    models_directory_patch = patch("emmo.bucket.base.MODELS_DIRECTORY", models_directory)
    models_directory_patch = patch("emmo.bucket.utils.MODELS_DIRECTORY", models_directory)

    models_directory_patch.start()

    try:
        yield
    finally:
        models_directory_patch.stop()


@contextlib.contextmanager
def should_pull_ctx(already_exists: bool, user_wants_to_download: bool) -> Iterator[MagicMock]:
    """Context manager to mock click.confirm and get_model_local_directory for should_pull."""
    click_confirm_patch = patch(
        "emmo.bucket.model_puller.click.confirm", return_value=user_wants_to_download
    )
    get_model_local_directory_patch = patch(
        "emmo.bucket.model_puller.get_local_path", return_value=MagicMock(spec=Path)
    )
    click_confirm = click_confirm_patch.start()
    local_directory = get_model_local_directory_patch.start()
    local_directory.return_value.exists.return_value = already_exists

    try:
        yield click_confirm
    finally:
        click_confirm_patch.stop()
        get_model_local_directory_patch.stop()


@pytest.mark.parametrize(
    ("model_uri", "expected_puller_cls"),
    [
        ("gs://biondeep-models/emmo/binding_predictor/my_model", DefaultModelPuller),
        ("s3://biondeep-models/emmo/binding_predictor/my_model", DefaultModelPuller),
    ],
)
def test_get_model_puller(model_uri: str, expected_puller_cls: type[BaseModelPuller]) -> None:
    """Ensure get_model_puller factory is working as expected."""
    assert isinstance(get_model_puller(model_uri), expected_puller_cls)


@pytest.mark.parametrize(
    "model_uri",
    [
        "gs://biondeep-models/emmo/binding_predictor/my_model_to_pull",
        "s3://biondeep-models/emmo/binding_predictor/my_model_to_pull",
    ],
)
def test_default_model_puller(models_directory: Path, model_uri: str) -> None:
    """Ensure the DefaultModelPuller is working as expected."""
    create_common_model_artifacts(CloudPath(model_uri))
    expected_local_directory = models_directory / "binding_predictor" / "my_model_to_pull"
    expected_file_paths = set(get_common_model_artifact_file_paths(expected_local_directory))

    with pull_ctx(models_directory):
        puller = DefaultModelPuller()
        puller.pull(model_uri)

    assert {fp for fp in expected_local_directory.rglob("*") if fp.is_file()} == expected_file_paths


@pytest.mark.parametrize("puller_cls", [DefaultModelPuller])
@pytest.mark.parametrize(
    ("already_exists", "force", "user_wants_to_download", "expected_output"),
    [
        (True, True, True, True),
        (True, True, False, True),
        (True, False, True, True),
        (True, False, False, False),
        (False, True, True, True),
        (False, True, False, True),
        (False, False, True, True),
        (False, False, False, True),
    ],
)
def test_should_pull(
    puller_cls: type[BaseModelPuller],
    already_exists: bool,
    force: bool,
    user_wants_to_download: bool,
    expected_output: bool,
) -> None:
    """Ensure the should_pull method is working as expected."""
    with should_pull_ctx(already_exists, user_wants_to_download) as click_confirm:
        puller = puller_cls()
        shoud_ask_user = already_exists and not force

        assert puller.should_pull("dummy_uri", force) is expected_output
        assert click_confirm.called is shoud_ask_user
