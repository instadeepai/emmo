"""Define classes to handle the pushing of the different model types in the Google/AWS bucket."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from pathlib import Path
from typing import Iterable

import click

from emmo.bucket.io import upload_to_directory
from emmo.bucket.utils import get_bucket_path
from emmo.utils import logger

log = logger.get(__name__)


class BaseModelPusher(ABC):
    """Base abstract class to define interface for model pusher."""

    def should_push(self, model_path: Path, force: bool) -> bool:
        """Indicates if the model should be pushed or not."""
        bucket_path = get_bucket_path(model_path)
        return not (
            bucket_path.exists()
            and not force
            and not click.confirm(
                f"The model is already available in the bucket at '{bucket_path}'. "
                f"Do you want to push it anyway?",
                default=True,
            )
        )

    def push(self, model_path: Path, **kwargs: str) -> None:
        """Main method to push the files linked to a model."""
        files_to_upload = self.get_files_to_upload(model_path, **kwargs)
        for fp in files_to_upload:
            bucket_directory_path = get_bucket_path(fp).parent
            upload_to_directory(fp, bucket_directory_path, force=True, verbose=True)

    @abstractmethod
    def get_files_to_upload(self, model_path: Path, **kwargs: str) -> Iterable[Path]:
        """Retrieve the list of files to upload.

        It must be implemented in child classes.
        """
        pass


class DefaultModelPusher(BaseModelPusher):
    """Default class to use to push model."""

    def get_files_to_upload(self, model_path: Path, **kwargs: str) -> Iterable[Path]:
        """The default pusher is pushing all the files in the model directory and subdirectories."""
        return {fp for fp in model_path.rglob("*") if fp.is_file()}


def get_model_pusher(model_path: Path) -> BaseModelPusher:
    """Retrieve the model pusher instance."""
    pusher_cls: type[BaseModelPusher] = DefaultModelPusher
    log.debug(f"{pusher_cls.__name__} used to push the model {model_path.name}")
    return pusher_cls()
