"""Define classes to handle the pulling of the different model types in the Google/AWS bucket."""
from __future__ import annotations

from abc import ABC
from abc import abstractmethod
from typing import Iterable

import click
from cloudpathlib import CloudPath

from emmo.bucket.io import download_to_directory
from emmo.bucket.utils import get_local_path
from emmo.constants import MODELS_DIRECTORY
from emmo.utils import logger

log = logger.get(__name__)


class BaseModelPuller(ABC):
    """Base abstract class to define interface for model puller."""

    def should_pull(self, model_uri: str, force: bool) -> bool:
        """Check if the model should be pulled."""
        local_model_directory = get_local_path(model_uri)

        return not (
            local_model_directory.exists()
            and not force
            and not click.confirm(
                f"The model {local_model_directory.relative_to(MODELS_DIRECTORY)} already exists "
                f"locally. It will be overwritten by the model {model_uri}. Do you confirm you "
                "want to download this model?"
            )
        )

    def pull(self, model_uri: str, **kwargs: str) -> None:
        """Main method to pull the files linked to a model."""
        files_to_download = self.get_files_to_download(CloudPath(model_uri), **kwargs)
        for fp in files_to_download:
            local_directory_path = get_local_path(fp).parent
            download_to_directory(local_directory_path, fp, force=True, verbose=True)

    @abstractmethod
    def get_files_to_download(self, model_uri: CloudPath, **kwargs: str) -> Iterable[str]:
        """Retrieve the list of files to download.

        It must be implemented in child classes.
        """
        pass


class DefaultModelPuller(BaseModelPuller):
    """Default class to use to pull a model."""

    def get_files_to_download(self, model_uri: CloudPath, **kwargs: str) -> Iterable[str]:
        """Retrieve the list of files to download."""
        return {str(fp) for fp in model_uri.rglob("*") if fp.is_file()}


def get_model_puller(model_uri: str) -> BaseModelPuller:
    """Retrieve the model puller instance."""
    puller_cls: type[BaseModelPuller] = DefaultModelPuller
    log.debug(f"{puller_cls.__name__} used to pull the model {model_uri}")
    return puller_cls()
