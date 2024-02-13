"""Test cases for module bucket I/O functions."""
from __future__ import annotations

import time
from pathlib import Path

import pytest
from cloudpathlib import CloudPath
from cloudpathlib.exceptions import OverwriteNewerCloudError

from emmo.bucket.io import download_to_directory
from emmo.bucket.io import upload_to_directory


@pytest.fixture()
def tmp_directory(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Fixture used to define a temporary directory used to store test files."""
    return tmp_path_factory.mktemp("test_bucket_io")


@pytest.mark.parametrize("cloud_provider", ["gs", "s3"])
def test_upload_local_file_not_exist_error(tmp_directory: Path, cloud_provider: str) -> None:
    """Ensure 'upload_to_directory' raises FileNotFoundError if 'local_path' does not exist."""
    local_path = tmp_directory / "non_existing_file.txt"

    with pytest.raises(FileNotFoundError):
        upload_to_directory(
            local_path=local_path,
            bucket_directory_path=f"{cloud_provider}://dummy-bucket/directory",
        )


@pytest.mark.parametrize("cloud_provider", ["gs", "s3"])
def test_upload_local_directory_bucket_file_error(tmp_directory: Path, cloud_provider: str) -> None:
    """Ensure 'upload_to_directory' raises NotADirectoryError.

    This error should be raised if 'bucket_directory_path' is a file.
    """
    expected_msg = "already exists but is not a directory."
    local_path = tmp_directory / "directory"
    local_path.mkdir()

    bucket_directory_path = f"{cloud_provider}://dummy-bucket/directory/file_in_bucket.txt"
    CloudPath(bucket_directory_path).touch()

    with pytest.raises(NotADirectoryError, match=expected_msg):
        upload_to_directory(local_path=local_path, bucket_directory_path=bucket_directory_path)


@pytest.mark.parametrize("cloud_provider", ["gs", "s3"])
def test_upload_no_force(tmp_directory: Path, cloud_provider: str) -> None:
    """Ensure 'upload_to_directory' raises OverwriteNewerCloudError.

    This error should be raised if a file already exists in the bucket and force=False.
    """
    file_name = "file.txt"
    local_path = tmp_directory / file_name
    local_path.touch()

    bucket_directory_path = CloudPath(f"{cloud_provider}://dummy-bucket/directory")
    (bucket_directory_path / file_name).touch()

    with pytest.raises(OverwriteNewerCloudError):
        upload_to_directory(
            local_path=local_path, bucket_directory_path=bucket_directory_path, force=False
        )


@pytest.mark.parametrize("cloud_provider", ["gs", "s3"])
def test_upload_local_file(tmp_directory: Path, cloud_provider: str) -> None:
    """Ensure 'download_to_directory' is working as expected."""
    file_name = "my_file.txt"
    local_path = tmp_directory / file_name
    local_path.touch()

    bucket_directory_path = f"{cloud_provider}://dummy-bucket/directory"

    upload_to_directory(local_path=local_path, bucket_directory_path=bucket_directory_path)

    assert (CloudPath(bucket_directory_path) / file_name).exists()


@pytest.mark.parametrize("cloud_provider", ["gs", "s3"])
@pytest.mark.parametrize("force", [True, False])
def test_upload_with_subdirectories_already_exist(
    tmp_directory: Path, force: bool, cloud_provider: str
) -> None:
    """Ensure 'download_to_directory' function is working as expected.

    The function is tested for a directory with subdirectories and a file that already exists.
    """
    _create_files(tmp_directory)

    bucket_directory_path = CloudPath(f"{cloud_provider}://dummy-bucket/my_directory")
    cloud_path = bucket_directory_path / tmp_directory.name

    already_exist_fp = cloud_path / "subdirectory1" / "file1.txt"
    already_exist_fp.touch()
    modified_time = already_exist_fp.stat().st_mtime

    expected_files = {
        cloud_path / "file1.txt",
        already_exist_fp,
        cloud_path / "subdirectory1" / "file2.txt",
        cloud_path / "subdirectory2" / "file1.txt",
        cloud_path / "subdirectory1" / "subdirectory" / "file1.txt",
    }

    # the upload could be too fast and 'st_mtime' won't change even with force=True
    time.sleep(0.01)

    upload_to_directory(
        local_path=tmp_directory, bucket_directory_path=bucket_directory_path, force=force
    )

    is_overwritten = modified_time != already_exist_fp.stat().st_mtime

    print({fp for fp in cloud_path.rglob("*") if fp.is_file()})
    assert {fp for fp in cloud_path.rglob("*") if fp.is_file()} == expected_files
    assert is_overwritten is force


@pytest.mark.parametrize("cloud_provider", ["gs", "s3"])
def test_download_bucket_file_not_exist_error(tmp_directory: Path, cloud_provider: str) -> None:
    """Ensure 'download_to_directory' raises FileNotFoundError if 'bucket_path' does not exist."""
    bucket_path = CloudPath(f"{cloud_provider}://dummy-bucket/directory/non_existing-file.txt")

    with pytest.raises(FileNotFoundError):
        download_to_directory(
            local_directory_path=tmp_directory,
            bucket_path=bucket_path,
        )


@pytest.mark.parametrize("cloud_provider", ["gs", "s3"])
@pytest.mark.parametrize("bucket_relative_path", ["directory", "directory/file_in_bucket.txt"])
def test_download_bucket_directory_local_file_error(
    tmp_directory: Path, cloud_provider: str, bucket_relative_path: str
) -> None:
    """Ensure 'download_to_directory' raises ValueError if 'local_directory_path' is a file."""
    expected_msg = "already exists but is not a directory."
    local_directory_path = tmp_directory / "file.txt"
    local_directory_path.touch()

    bucket_path = CloudPath(f"{cloud_provider}://dummy-bucket/{bucket_relative_path}")
    bucket_path.touch()

    with pytest.raises(NotADirectoryError, match=expected_msg):
        download_to_directory(local_directory_path=local_directory_path, bucket_path=bucket_path)


@pytest.mark.parametrize("cloud_provider", ["gs", "s3"])
def test_download_no_force(tmp_directory: Path, cloud_provider: str) -> None:
    """Ensure 'download_to_directory' raises FileExistsError.

    This error should be raised if a local file already exist and force=False."""
    file_name = "file.txt"
    local_file_path = tmp_directory / file_name
    local_file_path.touch()

    bucket_path = CloudPath(f"{cloud_provider}://dummy-bucket/directory") / file_name
    bucket_path.touch()

    with pytest.raises(FileExistsError):
        download_to_directory(
            local_directory_path=tmp_directory,
            bucket_path=bucket_path,
        )


@pytest.mark.parametrize("cloud_provider", ["gs", "s3"])
def test_download_bucket_file_local_directory(tmp_directory: Path, cloud_provider: str) -> None:
    """Ensure 'download_to_directory' is working as expected if 'bucket_path' is a file."""
    bucket_path = CloudPath(f"{cloud_provider}://dummy-bucket/directory/file.txt")
    bucket_path.touch()

    expected_local_path = tmp_directory / bucket_path.name

    download_to_directory(local_directory_path=tmp_directory, bucket_path=bucket_path)

    assert expected_local_path.exists()


@pytest.mark.parametrize("cloud_provider", ["gs", "s3"])
@pytest.mark.parametrize("force", [True, False])
def test_download_subdirectories_already_exist(
    tmp_directory: Path, cloud_provider: str, force: bool
) -> None:
    """Ensure 'download_to_directory' function is working as expected.

    The function is tested for a directory with subdirectories and a file that already exists.
    """
    bucket_directory = CloudPath(f"{cloud_provider}://dummy-bucket/to_download")
    local_path = tmp_directory / "to_download"

    _create_files(bucket_directory)

    already_exist_fp = local_path / "subdirectory1" / "file1.txt"
    already_exist_fp.parent.mkdir(parents=True)
    already_exist_fp.touch()
    modified_time = already_exist_fp.stat().st_mtime

    expected_files = {
        local_path / "file1.txt",
        already_exist_fp,
        local_path / "subdirectory1" / "file2.txt",
        local_path / "subdirectory2" / "file1.txt",
        local_path / "subdirectory1" / "subdirectory" / "file1.txt",
    }

    # the download could be too fast and 'st_mtime' won't change even with force=True
    time.sleep(0.01)

    download_to_directory(
        local_directory_path=tmp_directory, bucket_path=bucket_directory, force=force
    )

    is_overwritten = modified_time != already_exist_fp.stat().st_mtime

    assert {fp for fp in local_path.rglob("*") if fp.is_file()} == expected_files
    assert is_overwritten is force


def _create_files(path: Path | CloudPath) -> None:
    """Create files and subdirectories to use for the tests."""
    (path / "file1.txt").touch()
    (path / "subdirectory1").mkdir()
    (path / "subdirectory2").mkdir()
    (path / "subdirectory1" / "subdirectory").mkdir()
    (path / "subdirectory1" / "file1.txt").touch()
    (path / "subdirectory1" / "file2.txt").touch()
    (path / "subdirectory2" / "file1.txt").touch()
    (path / "subdirectory1" / "subdirectory" / "file1.txt").touch()
