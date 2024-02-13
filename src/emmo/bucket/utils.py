"""Module to define utils functions regarding bucket operations."""
from __future__ import annotations

import os
from pathlib import Path

from cloudpathlib import CloudPath

from emmo.constants import BUCKET_PREFIXES
from emmo.constants import DATA_BUCKET_NAME
from emmo.constants import DATA_DIRECTORY
from emmo.constants import GS_BUCKET_PREFIX
from emmo.constants import MODELS_BUCKET_NAME
from emmo.constants import MODELS_DIRECTORY
from emmo.constants import S3_BUCKET_PREFIX


def remove_bucket_prefix(uri: str) -> str:
    """Remove the bucket prefix and the bucket name from the file path.

    Args:
        uri: uri to clean

    >>> remove_bucket_prefix("gs://biondeep-models/presentation/first_model")
    'presentation/first_model'

    >>> remove_bucket_prefix("gs://random-bucket/directory/test.csv")
    'directory/test.csv'

    >>> remove_bucket_prefix("s3://biondeep-data/test.csv")
    'test.csv'
    """
    return os.path.join(*uri.split("/")[3:])


def get_local_path(bucket_path: str | CloudPath) -> Path:
    """Get the local path corresponding to the input bucket path.

    - for bucket 'biondeep-models': use MODELS_DIRECTORY as prefix
    - for bucket 'biondeep-data': use DATA_DIRECTORY as prefix

    The 'emmo' subdirectory in the models' bucket corresponds to the local 'models' directory.

    Raises:
        ValueError: if the bucket_path does not start by GS_BUCKET_PREFIX or S3_BUCKET_PREFIX or the
            bucket used is not known (i.e. not DATA_BUCKET_NAME or MODELS_BUCKET_NAME)
    """
    bucket_path = str(bucket_path)
    if not bucket_path.startswith(BUCKET_PREFIXES):
        raise ValueError(
            f"The bucket path ({bucket_path}) provided does not start with a known bucket prefix."
            f"Currently known bucket prefix are '{GS_BUCKET_PREFIX}' and '{S3_BUCKET_PREFIX}'"
        )

    relative_local_path = remove_bucket_prefix(bucket_path)
    if DATA_BUCKET_NAME in bucket_path:
        local_path = DATA_DIRECTORY / relative_local_path

    elif MODELS_BUCKET_NAME in bucket_path:
        if relative_local_path.startswith("emmo/"):
            relative_local_path = relative_local_path.split("/", 1)[1]

        local_path = MODELS_DIRECTORY / relative_local_path

    else:
        raise ValueError(
            f"The bucket used is unknown. Currently known buckets are '{DATA_BUCKET_NAME}' and "
            f"'{MODELS_BUCKET_NAME}'"
        )

    return local_path


def get_bucket_path(local_path: Path) -> CloudPath:
    """Get the bucket path corresponding to a file/directory in the local 'data' or 'models' folder.

    The local 'models' directory corresponds to the 'emmo' subdirectory in the models' bucket.

    Args:
        local_path: path to the file/directory in the local directory
    """
    if DATA_DIRECTORY in local_path.parents:
        bucket_name = DATA_BUCKET_NAME
        bucket_file_path = str(local_path.relative_to(DATA_DIRECTORY))

    elif MODELS_DIRECTORY in local_path.parents:
        bucket_name = MODELS_BUCKET_NAME
        bucket_file_path = str(Path("emmo") / local_path.relative_to(MODELS_DIRECTORY))

    else:
        raise ValueError(
            f"Only paths to files/directories in the local 'data' ({DATA_DIRECTORY}) or 'models' "
            f"({MODELS_DIRECTORY}) folder are supported."
        )

    return CloudPath(f"{GS_BUCKET_PREFIX}{bucket_name}") / bucket_file_path
