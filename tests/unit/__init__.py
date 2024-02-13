"""Package to define unit tests."""
from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
from unittest.mock import patch


@contextlib.contextmanager
def bucket_ctx(tmp_directory: Path) -> Iterator[None]:
    """Context manager to patch the DATA_DIRECTORY variable in emmo/bucket/base.py.

    We patch this variables to tmp_directory to be able to use .relative_to().
    """
    base_data_directory_patch = patch("emmo.bucket.base.DATA_DIRECTORY", tmp_directory)

    base_data_directory_patch.start()

    try:
        yield
    finally:
        base_data_directory_patch.stop()
