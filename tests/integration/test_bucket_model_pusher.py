"""Test cases for the classes defined in biondeep/bucket/model_pusher.py."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import pytest
from cloudpathlib import CloudPath

from emmo.bucket.model_pusher import BaseModelPusher
from emmo.bucket.model_pusher import DefaultModelPusher
from emmo.bucket.model_pusher import get_model_pusher
from emmo.constants import MODELS_DIRECTORY
from tests.helpers import create_common_model_artifacts
from tests.helpers import get_common_model_artifact_file_paths


@contextlib.contextmanager
def push_ctx(models_directory: Path, *bucket_path_to_remove: CloudPath) -> Iterator[None]:
    """Context manager used to mock the MODELS_DIRECTORY to use a tmp directory.

    It also cleans the bucket path to avoid any issue if we want to try to push
    to the same directory multiple times (e.g. with 2 different 'ckpt_name').
    """
    models_directory_patch = patch("emmo.bucket.base.MODELS_DIRECTORY", models_directory)
    models_directory_patch = patch("emmo.bucket.utils.MODELS_DIRECTORY", models_directory)

    models_directory_patch.start()

    try:
        yield
    finally:
        models_directory_patch.stop()
        for bucket_path in bucket_path_to_remove:
            bucket_path.rmtree()


@pytest.mark.parametrize(
    ("model_path", "expected_pusher_cls"),
    [
        (MODELS_DIRECTORY / "binding_predictor" / "my_model", DefaultModelPusher),
        (MODELS_DIRECTORY / "cleavage" / "my_model", DefaultModelPusher),
    ],
)
def test_get_model_pusher(model_path: Path, expected_pusher_cls: type[BaseModelPusher]) -> None:
    """Ensure get_model_pusher factory is working as expected."""
    assert isinstance(get_model_pusher(model_path), expected_pusher_cls)


def test_default_model_pusher(
    models_directory: Path,
) -> None:
    """Ensure the DefaultModelPusher is working as expected."""
    bucket_model_path = CloudPath("gs://biondeep-models/emmo/binding_predictor/my_model")
    expected_files = set(get_common_model_artifact_file_paths(bucket_model_path))

    with push_ctx(models_directory, bucket_model_path):
        model_path = models_directory / "binding_predictor" / "my_model"
        create_common_model_artifacts(model_path)

        model_pusher = DefaultModelPusher()
        model_pusher.push(model_path)

        assert {fp for fp in bucket_model_path.rglob("*") if fp.is_file()} == expected_files


@contextlib.contextmanager
def should_push_ctx(already_exists: bool, user_wants_to_upload: bool) -> Iterator[MagicMock]:
    """Context manager used to mock the useful resources for should_push method."""
    click_confirm_patch = patch(
        "emmo.bucket.model_pusher.click.confirm", return_value=user_wants_to_upload
    )
    get_bucket_path_patch = patch(
        "emmo.bucket.model_pusher.get_bucket_path", return_value=MagicMock(spec=CloudPath)
    )
    click_confirm = click_confirm_patch.start()
    bucket_directory = get_bucket_path_patch.start()
    bucket_directory.return_value.exists.return_value = already_exists

    try:
        yield click_confirm
    finally:
        click_confirm_patch.stop()
        get_bucket_path_patch.stop()


@pytest.mark.parametrize("pusher_cls", [DefaultModelPusher])
@pytest.mark.parametrize(
    ("already_exists", "force", "user_wants_to_upload", "expected_output"),
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
def test_should_push(
    pusher_cls: type[BaseModelPusher],
    already_exists: bool,
    force: bool,
    user_wants_to_upload: bool,
    expected_output: bool,
) -> None:
    """Ensure the should_push method is working as expected."""
    model_path = MODELS_DIRECTORY / "dummy_name"
    pusher = pusher_cls()
    with should_push_ctx(already_exists, user_wants_to_upload) as click_confirm:
        should_ask_user = already_exists and not force
        assert pusher.should_push(model_path, force) is expected_output
        assert should_ask_user is click_confirm.called
