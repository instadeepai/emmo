"""Command lines to pull/push data from/to bucket (GCP or S3)."""
from __future__ import annotations

from pathlib import Path

import click
from cloudpathlib import CloudPath

from emmo.bucket.model_puller import get_model_puller
from emmo.bucket.model_pusher import get_model_pusher
from emmo.constants import AVAILABLE_MODEL_DIRECTORIES
from emmo.utils.click import callback


@click.command()
@click.option(
    "--model_name",
    "-n",
    required=True,
    type=CloudPath,
    callback=callback.abort_if_not_exists,
    help=(
        "Folder of the model to pull. You must provide the full URI (e.g. "
        "gs://biondeep-models/emmo/binding_predictor/my_model)"
    ),
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force overwriting the model folder if it already exists locally.",
)
def pull_model(
    model_name: str,
    force: bool,
) -> None:
    """Pull a model from the GS models' bucket.

    The model will be pulled from the 'emmo' subdirectory in the bucket if available.
    """
    puller = get_model_puller(model_name)
    if puller.should_pull(model_name, force):
        puller.pull(model_name)


@click.command()
@click.option(
    "--model_name",
    "-n",
    type=str,
    required=True,
    help=("Model to use: it can be the name of the folder where the model is saved or the path."),
)
@click.option(
    "--force",
    "-f",
    is_flag=True,
    help="Force overwrite the model bucket folder if it already exists.",
)
def push_model(
    model_name: str,
    force: bool,
) -> None:
    """Push a model into the 'biondeep-models' bucket on GS.

    The model will be pushed to the 'emmo' subdirectory in the bucket.
    """
    model_path = get_model_path_from_name(model_name)
    model_pusher = get_model_pusher(model_path)
    if model_pusher.should_push(model_path, force):
        model_pusher.push(model_path)


def get_model_path_from_name(model_name: str) -> Path:
    """Search for a model name in the available model directories and return the model path.

    Args:
        model_name: name of the model

    Raises:
        FileNotFoundError: if the model name was not found
        ValueError: if the model name is ambiguous
    """
    model_name = Path(model_name).name

    models_found = []
    for model_directory in AVAILABLE_MODEL_DIRECTORIES:
        model_path = model_directory / model_name
        if model_path.is_dir():
            models_found.append(model_path)

    if not models_found:
        model_directories = " | ".join(str(directory) for directory in AVAILABLE_MODEL_DIRECTORIES)
        raise FileNotFoundError(
            f"Model folder with name: {model_name} not found in any of the available model "
            f"directory folders: {model_directories}"
        )

    if len(models_found) > 1:
        duplicate_models = " | ".join(str(model_path) for model_path in models_found)
        raise ValueError(f"The following models were found with the same name: {duplicate_models}")

    return models_found[0]
