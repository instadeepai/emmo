"""Configuration file for shared fixtures used by pytest tests."""
from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path

import pytest
from _pytest.monkeypatch import MonkeyPatch
from cloudpathlib import implementation_registry
from cloudpathlib.local import local_gs_implementation
from cloudpathlib.local import local_s3_implementation
from cloudpathlib.local import LocalGSClient
from cloudpathlib.local import LocalS3Client


@pytest.fixture(scope="session")
def tmp_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture used to define a temporary directory.

    Used to store test files other than cached data or models.
    """
    return tmp_path_factory.mktemp("test")


@pytest.fixture()
def models_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture used to define a temporary directory for models.

    Use 'scope=function' (default one) to avoid any issue if several models have the same type +
    name.
    """
    return tmp_path_factory.mktemp("models")


@pytest.fixture(scope="session")
def cloudpathlib_cache_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture used to define a temporary directory for the bucket's local cache directory."""
    return tmp_path_factory.mktemp(".cache_bucket")


@contextlib.contextmanager
def s3_manager_ctx() -> Iterator[None]:
    """Context manager to temporarily set required S3 environment variables with dummy values."""
    old_environ = dict(os.environ)
    os.environ.update(
        {
            "AWS_ACCESS_KEY_ID": "dummy_key_id",
            "AWS_SECRET_ACCESS_KEY": "dummy_access_key",
            "AWS_ENDPOINT_URL": "dummy_endpoint",
        }
    )
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


@contextlib.contextmanager
def gs_manager_ctx() -> Iterator[None]:
    """Context manager to temporarily set GOOGLE_APPLICATION_CREDENTIALS to a dummy value."""
    old_environ = dict(os.environ)
    os.environ.update(
        {
            "GOOGLE_APPLICATION_CREDENTIALS": tempfile.NamedTemporaryFile().name,
            "GS_CREDENTIALS_ENCODED": "eyJwcml2YXRlX2tleSI6ICJkdW1teV9jcmVkZW50aWFscyJ9",
        }
    )
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


@contextlib.contextmanager
def cloudpathlib_cache_ctx(cloudpathlib_cache_directory: Path) -> Iterator[None]:
    """Context manager to temporarily set CLOUDPATHLIB_LOCAL_CACHE_DIR in the tmp directory."""
    old_environ = dict(os.environ)
    os.environ.update(
        {
            "CLOUDPATHLIB_LOCAL_CACHE_DIR": str(cloudpathlib_cache_directory),
        }
    )
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(old_environ)


@pytest.fixture(scope="session", autouse=True)
def monkey_session() -> Iterator[MonkeyPatch]:
    """Fixture to use monkeypatch at session scope.

    Inspiration from https://stackoverflow.com/a/53963978/11194702.
    """
    monkey_patch = MonkeyPatch()
    yield monkey_patch
    monkey_patch.undo()


@pytest.fixture(scope="session", autouse=True)
def _bucket_client_mock(
    monkey_session: MonkeyPatch, cloudpathlib_cache_directory: Path
) -> Iterator[None]:
    """Fixture to mock the bucket clients for all tests.

    We mock connections as advised in
    https://cloudpathlib.drivendata.org/v0.9/testing_mocked_cloudpathlib/.
    """
    # Mock CloudPath dispatch
    monkey_session.setitem(implementation_registry, "s3", local_s3_implementation)
    monkey_session.setitem(implementation_registry, "gs", local_gs_implementation)

    try:
        with s3_manager_ctx(), gs_manager_ctx(), cloudpathlib_cache_ctx(
            cloudpathlib_cache_directory
        ):
            yield
    finally:
        LocalS3Client.reset_default_storage_dir()
        LocalGSClient.reset_default_storage_dir()


@pytest.fixture(scope="session")
def example_sequences() -> list[str]:
    """Test sequences."""
    return [
        "MADSRDPASD",
        "QMQHWKEQRAAQ",
        "KADVLT",
        "TGAGNPVGDKLN",
        "VITVGPRGPL",
        "LVQDVVFTDEMA",
        "HFDRERIP",
        "ERVVHAKGAG",
    ]


@pytest.fixture(scope="session")
def example_sequences_equal_length() -> list[str]:
    """Test sequences of equal_length."""
    return [
        "MADSRDPASD",
        "QMQHWKEQRA",
        "AQKADVLTTG",
        "AGNPVGDKLN",
        "VITVGPRGPL",
        "LVQDVVFTDE",
        "MAHFDRERIP",
        "ERVVHAKGAG",
    ]


@pytest.fixture(scope="module")
def expected_amino_acid_counts() -> list[int]:
    """Expected amino acid counts."""
    return [9, 0, 8, 4, 2, 7, 3, 2, 4, 4, 3, 2, 5, 5, 6, 2, 4, 9, 1, 0]


@pytest.fixture(scope="module")
def expected_amino_acid_frequencies() -> list[float]:
    """Expected amino acid frequencies."""
    return [
        0.1125,
        0.0,
        0.1,
        0.05,
        0.025,
        0.0875,
        0.0375,
        0.025,
        0.05,
        0.05,
        0.0375,
        0.025,
        0.0625,
        0.0625,
        0.075,
        0.025,
        0.05,
        0.1125,
        0.0125,
        0.0,
    ]
