"""Module used to define functions for remote I/O operations."""
from __future__ import annotations

from pathlib import Path

from cloudpathlib import CloudPath
from cloudpathlib.exceptions import OverwriteNewerCloudError

from emmo.utils import logger

log = logger.get(__name__)


def upload_to_directory(
    local_path: str | Path,
    bucket_directory_path: str | CloudPath,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Upload local file/directory to the bucket.

    The following use cases are handled:
    - 'local_path' is a file: the local filename is kept.
    - 'local_path' is a directory:
        - if force=False, only the missing files are uploaded
        - all the subdirectories are also uploaded.

    Raises:
        NotADirectoryError: 'bucket_directory_path' exists but is not a directory
        IsADirectoryError: a destination file path in the bucket already exists and is a directory
        FileNotFoundError: 'local_path' does not exist
        OverwriteNewerCloudError: a file with name 'local_path.name' in the bucket directory
            'bucket_directory_path' already exists and force=False
    """
    local_path = Path(local_path)
    bucket_directory_path = CloudPath(bucket_directory_path)

    if not local_path.exists():
        raise FileNotFoundError(f"The file or directory '{local_path}' does not exist locally.")

    if bucket_directory_path.exists() and not bucket_directory_path.is_dir():
        raise NotADirectoryError(
            f"Bucket path {bucket_directory_path} already exists but is not a directory."
        )

    if local_path.is_file():
        bucket_file_path = bucket_directory_path / local_path.name

        if bucket_file_path.is_dir():
            raise IsADirectoryError(
                f"Could not upload file {local_path} to bucket file path {bucket_file_path} "
                "since the latter exists and is a directory."
            )

        bucket_file_path.upload_from(local_path, force_overwrite_to_cloud=force)

        if verbose:
            log.info(f"{local_path} uploaded at {bucket_file_path}.")
    else:
        _upload_directory(
            local_path=local_path,
            bucket_directory_path=bucket_directory_path,
            force=force,
            verbose=verbose,
        )


def download_to_directory(
    local_directory_path: str | Path,
    bucket_path: str | CloudPath,
    force: bool = False,
    verbose: bool = False,
) -> None:
    """Download remote file/directory locally.

    The following use cases are handled:
    - 'bucket_path' is a file: the bucket filename is kept.
    - 'bucket_path' is a directory:
        - if force=False, only the missing files are downloaded.
        - all the subdirectories are also downloaded.

    Raises:
        NotADirectoryError: 'local_directory_path' exists but is not a directory
        IsADirectoryError: a local destination file path already exists and is a directory
        FileNotFoundError: 'bucket_path' does not exist.
        FileExistsError: a local destination file path already exists and force=False
    """
    local_directory_path = Path(local_directory_path)
    bucket_path = CloudPath(bucket_path)

    if not bucket_path.exists():
        raise FileNotFoundError(f"The file or directory '{bucket_path}' does not exists.")

    if local_directory_path.exists() and not local_directory_path.is_dir():
        raise NotADirectoryError(
            f"Local path {local_directory_path} already exists but is not a directory."
        )

    # Ensure that 'local_directory_path' exists as a directory such that 'download_to()' downloads
    # the bucket file/directory into this directory
    local_directory_path.mkdir(exist_ok=True, parents=True)

    if bucket_path.is_file():
        local_file_path = local_directory_path / bucket_path.name

        if local_file_path.is_dir():
            raise IsADirectoryError(
                f"Could not download file {bucket_path} to local file path {local_file_path} "
                "since the latter exists and is a directory."
            )

        if local_file_path.is_file() and not force:
            raise FileExistsError(
                f"Local file {local_file_path} already exist. Set force=True to overwrite it."
            )

        bucket_path.download_to(local_directory_path)

        if verbose:
            log.info(f"{bucket_path} downloaded at {local_file_path}.")

    else:
        _download_directory(
            local_directory_path=local_directory_path,
            bucket_path=bucket_path,
            force=force,
            verbose=verbose,
        )


def _upload_directory(
    local_path: Path, bucket_directory_path: CloudPath, force: bool, verbose: bool
) -> None:
    """Upload a directory (and subdirectories) to the bucket."""
    for local_path_ in local_path.iterdir():
        try:
            upload_to_directory(
                local_path=local_path_,
                bucket_directory_path=bucket_directory_path / local_path.name,
                force=force,
                verbose=verbose,
            )
        except OverwriteNewerCloudError:
            bucket_file_path = bucket_directory_path / local_path.name / local_path_.name
            log.warning(
                f"File {bucket_file_path} already exists in the bucket. "
                "Please set force=True to overwrite it."
            )


def _download_directory(
    local_directory_path: Path, bucket_path: CloudPath, force: bool, verbose: bool
) -> None:
    """Download a directory (and subdirectories) locally."""
    for bucket_path_ in bucket_path.iterdir():
        try:
            download_to_directory(
                local_directory_path=local_directory_path / bucket_path.name,
                bucket_path=bucket_path_,
                force=force,
                verbose=verbose,
            )
        except FileExistsError as file_exists_error:
            log.warning(str(file_exists_error))
