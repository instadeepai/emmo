"""Selection for deconvolution models and motifs for individual alleles or groups.

This module provides a function to select deconvolution models and motifs for individual alleles or
groups. The selection can be configured by providing a YAML file that defines the selected models
and motifs. The YAML file can also be used to remove alleles or groups from the selection.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any
from typing import TypedDict

from cloudpathlib import AnyPath
from cloudpathlib import CloudPath

from emmo.io.file import load_csv
from emmo.io.file import load_yml
from emmo.io.file import Openable
from emmo.io.output import find_deconvolution_results
from emmo.models.deconvolution import DeconvolutionModel
from emmo.utils import logger

log = logger.get(__name__)


class SelectedModel(TypedDict, total=False):
    """Dictionary defining the selected model and motif for a specific allele."""

    classes: int
    motif: int
    model_path: Openable
    comment: str
    model: DeconvolutionModel
    effective_peptide_number: float
    motif_weight: float


def get_best_run_details(
    model_path: Openable,
    score_column: str | None = None,
) -> dict[str, int | float]:
    """Get the details of the best run of a model.

    Parses the `runs.csv` file of a model and returns the details of the best run.

    Args:
        model_path: Path to the model directory.
        score_column: Name of the column to use for selecting the best run. If None, the column
            'log_likelihood' is used for MHC1 models and 'score' for MHC2 models.

    Returns:
        Dictionary with the details of the best run.
    """
    path_runs = AnyPath(model_path) / "runs.csv"
    df_runs = load_csv(path_runs)

    if score_column is None:
        # both MHC1 and MHC2 deconvolution models have a column "log_likelihood", MHC2 models also
        # have a column "score", which is used for MHC2 model selection
        score_column = "score" if "score" in df_runs.columns else "log_likelihood"
    elif score_column not in df_runs.columns:
        raise ValueError(f"Column '{score_column}' not found in {path_runs}")

    best_run = df_runs.loc[df_runs[score_column].idxmax()]

    run_details: dict[str, int | float] = {"run": int(best_run.name) + 1}

    for col in df_runs.columns:
        if df_runs[col].dtype == "int64":
            run_details[col] = int(best_run[col])
        else:
            run_details[col] = float(best_run[col])

    return run_details


def select_deconvolution_models(
    models_directory: Path | CloudPath,
    selection_path: Path | CloudPath | None,
) -> dict[str, SelectedModel]:
    """Find deconvolution models in a directory and possibly override the default selection.

    For each allele (or another type of groups), the single motif in the deconvolution run with only
    one class is selected from the models found in 'models_directory'. Optionally, the selected
    model and motif can be overridden for individual alleles and alleles can be removed by providing
    a path to a YAML file ('selection_path') that defines these modifications.

    Args:
        models_directory: Path to the local or remote directory containing the deconvolution models.
        selection_path: Path to the local or remote YAML file containing the models and motifs to
            be selected.

    Returns:
        The selected models and motifs as a dictionary containing as keys the alleles (or groups)
        and as values dictionaries with keys 'classes', 'motif', and 'model_path'.
    """
    available_models = {
        (str(row["group"]), int(row["number_of_classes"])): row["model_path"]
        for _, row in find_deconvolution_results(models_directory).iterrows()
    }

    num_groups = len({group for group, _ in available_models})

    log.info(f"Found {len(available_models)} models covering {num_groups} alleles/groups")

    # by default, select motif from deconvolution runs with one class
    selected_models: dict[str, SelectedModel] = {
        str(group): {"classes": number_of_classes, "motif": 1, "model_path": model_path}
        for (group, number_of_classes), model_path in available_models.items()
        if number_of_classes == 1
    }

    # apply the custom model and motif selection from the YAML file
    if selection_path is not None:
        for group, selection in load_yml(selection_path).items():
            _select_alternative_models(group, selection, selected_models, available_models)

    log.info(f"Selected models for {len(selected_models)} alleles/groups")

    return selected_models


def _convert_to_int(value: Any) -> int:
    """Check if 'value' represents an integer and, if yes, return this integer.

    Args:
        value: Value to check.

    Raises:
        TypeError: If 'value' is not of type int, str, or float.

    Returns:
        The value converted to an int.
    """
    if isinstance(value, int):
        return value

    if isinstance(value, float):
        value_int = int(value)
        if float(value_int) == value:
            return value_int

    if isinstance(value, str):
        return int(value)

    raise TypeError(f"'value' must be of type int, str, or float, but is {type(value)}")


def _select_alternative_models(
    group: str,
    selection: dict[str, int | Openable],
    selected_models: dict[str, SelectedModel],
    available_models: dict[tuple[str, int], Openable],
) -> None:
    """Select alternative models and motifs as defined in the YAML file.

    Args:
        group: The allele/group.
        selection: The selected class number and motif parsed from the YAML file.
        selected_models: The currently selected  models and motifs for each allele/group.
        available_models: A dict with all available alleles and numbers of classes as keys and the
            corresponding model paths as values.

    Raises:
        ValueError: If keys 'classes' or 'motif' are missing in 'selection'.
        ValueError: If the value of 'motif' is smaller than 1.
        ValueError: If the value of 'classes' is smaller than that of 'motif'.
        ValueError: If the selected number of classes is not among the available models.
    """
    if not selection.get("keep", True):
        if group not in selected_models:
            log.warning(f"allele/group {group} not available for removal")
        else:
            del selected_models[group]
            log.info(f"Removed allele/group {group} from selection")
        return

    if "classes" not in selection or "motif" not in selection:
        raise ValueError(f"'classes' and 'motif' need to be selected for allele/group {group}")

    classes = _convert_to_int(selection["classes"])
    motif = _convert_to_int(selection["motif"])

    if motif < 1:
        raise ValueError(f"'motif' < 1 for allele/group {group}")
    if classes < motif:
        raise ValueError(f"'classes' < 'motif' for allele/group {group}")

    try:
        selected_models[group]["model_path"] = available_models[group, classes]
    except KeyError:
        raise ValueError(f"no model with {classes} classes found for allele/group {group}")

    selected_models[group]["classes"] = classes
    selected_models[group]["motif"] = motif
    if "comment" in selection:
        selected_models[group]["comment"] = str(selection["comment"])
