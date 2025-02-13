"""Command line tools for binding prediction."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import click
import pandas as pd
from cloudpathlib import AnyPath
from cloudpathlib import CloudPath

from emmo.constants import MHC2_ALPHA_COL
from emmo.constants import MHC2_BETA_COL
from emmo.constants import MHC2_BINDING_CORE_SIZE
from emmo.constants import MODELS_DIRECTORY
from emmo.io.file import load_csv
from emmo.io.file import load_yml
from emmo.io.file import Openable
from emmo.io.file import save_csv
from emmo.io.output import find_deconvolution_results
from emmo.models.deconvolution import DeconvolutionModelMHC2
from emmo.models.prediction import PredictorMHC2
from emmo.models.prediction import SelectedModel
from emmo.pipeline.background import Background
from emmo.utils import logger
from emmo.utils.alleles import parse_mhc2_allele_pair
from emmo.utils.click import arguments
from emmo.utils.click import callback
from emmo.utils.viz import plot_predictor_mhc2

log = logger.get(__name__)


@click.command()
@click.option(
    "--input_file",
    "-i",
    type=AnyPath,
    callback=callback.abort_if_not_exists,
    required=True,
    help="Path to remote or local input CSV file.",
)
@click.option(
    "--output_file",
    "-o",
    type=AnyPath,
    required=True,
    help="Local or remote file path to save the CSV file with predictions.",
)
@click.option(
    "--model",
    "-m",
    type=str,
    required=True,
    help=(
        f"Path to the model directory, name of a model in the models folder {MODELS_DIRECTORY}, "
        "or bucket path."
    ),
)
@click.option(
    "--model_name",
    "-n",
    type=str,
    required=False,
    default=None,
    help=(
        "The name of the model to be used as a prefix of the result columns. "
        "If this is not provided, the name of the model directory is used."
    ),
)
@arguments.peptide_and_allele_columns_mhc2
@click.option(
    "--force",
    is_flag=True,
    help=(
        "If this flag is added, the deconvolution is run and output is written "
        "even if the output directory already exists."
    ),
)
@click.option(
    "--disable_offset_weight",
    is_flag=True,
    help=(
        "If this flag is added, the weights for the possible offset positions of the binding core "
        "are not used, i.e., only the PSSM determines the score for each offset."
    ),
)
@click.option(
    "--length_scoring",
    is_flag=True,
    help=(
        "If this flag is added, include the result columns that use the probability in the length "
        "distribution as an additional term in the scoring function."
    ),
)
def predict_mhc2(
    input_file: Path | CloudPath,
    output_file: Path | CloudPath,
    model: str,
    model_name: str | None,
    peptide_column: str,
    allele_alpha_column: str,
    allele_beta_column: str,
    force: bool,
    disable_offset_weight: bool,
    length_scoring: bool,
) -> None:
    """Run the prediction for MHC2 peptides and alleles."""
    _check_output_file_path(output_file, force)

    model_path = _get_model_path(model)
    predictor = PredictorMHC2.load(model_path)
    model_name = model_name if model_name is not None else model_path.name

    df = load_csv(input_file)

    predictor.score_dataframe(
        df,
        peptide_column=peptide_column,
        allele_alpha_column=allele_alpha_column,
        allele_beta_column=allele_beta_column,
        column_prefix=model_name,
        score_length=length_scoring,
        pan_allelic="nearest",
        use_offset_weight=(not disable_offset_weight),
        inplace=True,
    )

    save_csv(df, output_file, index=False, force=True)


@click.command()
@click.option(
    "--input_directory",
    "-i",
    type=AnyPath,
    required=True,
    callback=callback.abort_if_not_exists,
    help="Path to the local or remote directory containing the deconvolution results.",
)
@click.option(
    "--cleavage_model_path",
    "-c",
    type=AnyPath,
    required=True,
    callback=callback.abort_if_not_exists,
    help="Path to the local or remote directory containing the cleavage model.",
)
@click.option(
    "--selection_path",
    "-s",
    type=AnyPath,
    required=False,
    default=None,
    help=(
        "Path to the local or remote YAML file containing the models and motifs to be selected"
        " for the predictor."
    ),
)
@click.option(
    "--output_directory",
    "-o",
    type=AnyPath,
    required=False,
    default=None,
    help=(
        "Local or remote output directory in which the predictor directory is saved. If this is "
        "not provided, the predictor is saved in subdirectory 'binding_predictor' of the default "
        "models directory within the repository directory."
    ),
)
@click.option(
    "--name_prefix",
    "-n",
    type=str,
    required=False,
    default=None,
    help=(
        "Prefix for the predictor directory. If this is not provided, the name of the input "
        "directory is used."
    ),
)
@click.option(
    "--recompute_ppms",
    "-r",
    is_flag=True,
    help=(
        "If this flag is set, the position probability matrices are recomputed from the core "
        "predictions in the deconvolution runs. The 'responsibilities.csv' file must be contained "
        "in the respective deconvolution models directories."
    ),
)
@click.option(
    "--plot",
    is_flag=True,
    help="If this flag is set, a summary of the predictor is plotted in the subdirectory 'plots'.",
)
def compile_predictor_mhc2(
    input_directory: Path | CloudPath,
    cleavage_model_path: Path | CloudPath,
    selection_path: Path | CloudPath | None,
    output_directory: Path | CloudPath | None,
    name_prefix: str | None,
    recompute_ppms: bool,
    plot: bool,
) -> None:
    """Compile an MHC2 predictor from deconvolution results.

    For each allele, the single motif in the deconvolution run with only one class is selected from
    the models found in 'input_directory'. Optionally, the selected motif can be overridden for
    individual alleles and alleles can be removed by providing a path to a YAML file
    ('selection_path') that defines these modifications.
    """
    if output_directory is None:
        output_directory = MODELS_DIRECTORY / "binding_predictor"

    if name_prefix is None:
        name_prefix = input_directory.name

    ppm_mode = "recomputed" if recompute_ppms else "direct"
    time_stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    predictor_name = f"{name_prefix}_{ppm_mode}_{time_stamp}"
    predictor_directory = output_directory / predictor_name

    # select the model either through the selection file or by using the default trained using only
    # one class
    selected_models = _select_models(input_directory, selection_path)

    # try to load the amino acid background distribution
    background_path = input_directory / "background.json"
    if background_path.is_file():
        background = Background.load(background_path)
        log.info(f"Loaded background distribution from {background_path}")
    else:
        background = None
        log.warning(
            f"Background distribution does not exist at '{background_path}', will select a "
            f"distribution from one of the deconvolution models"
        )

    predictor = PredictorMHC2.compile_from_selected_models(
        selected_models,
        cleavage_model_path,
        background,
        peptides_path=None,
        motif_length=MHC2_BINDING_CORE_SIZE,
        length_distribution=None,
        recompute_ppms_from_best_responsibility=recompute_ppms,
    )

    predictor.save(predictor_directory, force=False)
    log.info(f"Saved compiled MHC2 predictor at {predictor_directory}")

    if plot:
        log.info("Plotting MHC2 predictor ...")
        plot_predictor_mhc2(predictor, predictor_directory / "plots")


@click.command()
@click.option(
    "--input_file",
    "-i",
    type=AnyPath,
    callback=callback.abort_if_not_exists,
    required=True,
    help="Path to remote or local input CSV file.",
)
@click.option(
    "--output_file",
    "-o",
    type=AnyPath,
    required=True,
    help="Local or remote file path to save the CSV file with predictions.",
)
@click.option(
    "--models_directory",
    "-m",
    type=AnyPath,
    required=True,
    callback=callback.abort_if_not_exists,
    help="Path to the local or remote directory containing the deconvolution results.",
)
@click.option(
    "--selection_path",
    "-s",
    type=AnyPath,
    required=False,
    default=None,
    help=(
        "Path to the local or remote YAML file containing the models and motifs to be selected "
        "for the predictor."
    ),
)
@click.option(
    "--prefix",
    "-p",
    type=str,
    required=False,
    default="emmo",
    help=("The prefix of the result columns."),
)
@click.option(
    "--force",
    is_flag=True,
    help=(
        "If this flag is added, the deconvolution is run and output is written "
        "even if the output directory already exists."
    ),
)
def predict_from_deconvolution_models_mhc2(
    input_file: Path | CloudPath,
    output_file: Path | CloudPath,
    models_directory: Path | CloudPath,
    selection_path: Path | CloudPath | None,
    prefix: str,
    force: bool,
) -> None:
    """Run the prediction for MHC2 peptides and alleles using deconvolution models directly."""
    _check_output_file_path(output_file, force)

    # select the model either through the selection file or by using the default trained using only
    # one class
    selected_models = _select_models(models_directory, selection_path)

    for selection in selected_models.values():
        selection["model"] = DeconvolutionModelMHC2.load(selection["model_path"])
    log.info("Loaded all deconvolution models")

    df = load_csv(input_file)

    result_columns = [
        f"{prefix}_{col}"
        for col in (
            "score",
            "best_class",
            "best_offset",
            "assigned_to_selected_motif",
        )
    ]
    column_intersection = sorted(set(df.columns) & set(result_columns))
    if column_intersection:
        raise ValueError(f"Dataframe already has columns: {column_intersection}")

    log.info(f"Scoring dataframe with {len(df):,} rows ...")
    df[result_columns] = df.apply(
        _score_with_deconvolution_models,
        args=(selected_models,),
        axis=1,
        result_type="expand",
    )

    save_csv(df, output_file, index=False, force=True)
    log.info(f"Saved results at {output_file}")


def _check_output_file_path(output_file: Path | CloudPath, force: bool) -> None:
    """Check if the output file path already exists.

    Args:
        output_file: Path to the local or remote output file.
        force: Whether to overwrite the file if it already exists.

    Raises:
        FileExistsError: If the file already exists and force is False.
        ValueError: If the path points to an existing directory.
    """
    if output_file.exists():
        if not force:
            raise FileExistsError(
                f"the output file {output_file} already exists, use --force to overwrite"
            )
        elif output_file.is_dir():
            raise ValueError(f"the output path {output_file} exists and is a directory")


def _get_model_path(model: str) -> Path | CloudPath:
    """Return the model path and the name of the model directory.

    Args:
        model: The model path or name of the precompiled models in the models folder of the
            repository.

    Returns:
        The (possibly inferred) path to the model.
    """
    model_path = AnyPath(model)

    if not model_path.exists():
        # if path does not exist, try to find the precompiled model this will only work, if the
        # models folder has the correct location relative to the package
        model_path = MODELS_DIRECTORY / "binding_predictor" / model

    if not model_path.exists():
        raise ValueError(
            f"Was not able to identify location of model {model}. Make sure that the 'models' "
            "folder is reachable relative to the package location or provide or valid path "
            "(absolute or relative to working directory, or cloud path)."
        )
    elif not model_path.is_dir():
        raise ValueError(f"The model path {model_path} exists but is not a directory.")

    return model_path


def _select_models(
    models_directory: Path | CloudPath,
    selection_path: Path | CloudPath | None,
) -> dict[str, SelectedModel]:
    """Find deconvolution models in a directory and possibly override the default selection.

    For each allele, the single motif in the deconvolution run with only one class is selected from
    the models found in 'models_directory'. Optionally, the selected model and motif can be
    overridden for individual alleles and alleles can be removed by providing a path to a YAML file
    ('selection_path') that defines these modifications.

    Args:
        models_directory: Path to the local or remote directory containing the deconvolution models.
        selection_path: Path to the local or remote YAML file containing the models and motifs to
            be selected.

    Returns:
        The selected models and motifs as a dictionary containing as keys the alleles
        ('<alpha_chain>-<beta_chain>') and as values dictionaries with keys 'classes', 'motif', and
        'model_path'.
    """
    available_models = {
        (*parse_mhc2_allele_pair(row["group"]), row["number_of_classes"]): row["model_path"]
        for _, row in find_deconvolution_results(models_directory).iterrows()
    }

    num_allele_pairs = len(
        {(allele_alpha, allele_beta) for allele_alpha, allele_beta, _ in available_models}
    )

    log.info(f"Found {len(available_models)} models covering {num_allele_pairs} allele pairs")

    # by default, select motif from deconvolution runs with one class
    selected_models: dict[str, SelectedModel] = {
        f"{allele_alpha}-{allele_beta}": {
            "classes": number_of_classes,
            "motif": 1,
            "model_path": model_path,
        }
        for (allele_alpha, allele_beta, number_of_classes), model_path in available_models.items()
        if number_of_classes == 1
    }

    # apply the custom model and motif selection from the YAML file
    if selection_path is not None:
        for allele, selection in load_yml(selection_path).items():
            _select_alternative_models(allele, selection, selected_models, available_models)

    log.info(f"Selected models for {len(selected_models)} alleles")

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
    allele: str,
    selection: dict[str, int | Openable],
    selected_models: dict[str, SelectedModel],
    available_models: dict[tuple[str, str, int], Openable],
) -> None:
    """Select alternative models and motifs as defined in the YAML file.

    Args:
        allele: The allele (alpha and beta chain separated by a hyphen).
        selection: The selected class number and motif parsed from the YAML file.
        selected_models: The currently selected  models and motifs for each allele.
        available_models: A dict with all available alleles and numbers of classes as keys and the
            corresponding model paths as values.

    Raises:
        ValueError: If keys 'classes' or 'motif' are missing in 'selection'.
        ValueError: If the value of 'motif' is smaller than 1.
        ValueError: If the value of 'classes' is smaller than that of 'motif'.
        ValueError: If the selected number of classes is not among the available models.
    """
    allele_alpha, allele_beta = parse_mhc2_allele_pair(allele)
    allele = f"{allele_alpha}-{allele_beta}"

    if not selection.get("keep", True):
        if allele not in selected_models:
            log.warning(f"allele {allele} not available for removal")
        else:
            del selected_models[allele]
            log.info(f"Removed allele {allele} from selection")
        return

    if "classes" not in selection or "motif" not in selection:
        raise ValueError(f"'classes' and 'motif' need to be selected for allele {allele}")

    classes = _convert_to_int(selection["classes"])
    motif = _convert_to_int(selection["motif"])

    if motif < 1:
        raise ValueError(f"'motif' < 1 for allele {allele}")
    if classes < motif:
        raise ValueError(f"'classes' < 'motif' for allele {allele}")

    try:
        selected_models[allele]["model_path"] = available_models[allele_alpha, allele_beta, classes]
    except KeyError:
        raise ValueError(f"no model with {classes} classes found for allele {allele}")

    selected_models[allele]["classes"] = classes
    selected_models[allele]["motif"] = motif
    if "comment" in selection:
        selected_models[allele]["comment"] = str(selection["comment"])


def _score_with_deconvolution_models(
    row: pd.Series,
    selected_models: dict[str, SelectedModel],
) -> tuple[float, str, str, bool]:
    """Score a row with the deconvolution model selected for the respective allele.

    Args:
        row: The row in the dataframe to score containing 'peptide', 'allele_alpha', and
            'allele_beta'.
        selected_models: The selected models and motifs for each allele.

    Returns:
        The total score given the respective devolution model, the best class, the best offset (for
        this class), and whether the best class is equal to the motif for the deconvolution model.
    """
    peptide = row["peptide"]
    allele_alpha = row[MHC2_ALPHA_COL]
    allele_beta = row[MHC2_BETA_COL]
    allele = f"{allele_alpha}-{allele_beta}"

    if allele not in selected_models:
        return float("nan"), "", "", False

    model = selected_models[allele]["model"]
    selected_motif = str(selected_models[allele]["motif"])

    score, best_class, best_offset = model.score_peptide(peptide, include_flat=True)

    # convert best_offset to str such that it is written to the csv file without a decimal point
    return score, best_class, str(best_offset), best_class == selected_motif
