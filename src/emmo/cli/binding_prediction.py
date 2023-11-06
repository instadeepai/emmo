"""Command line tools for binding prediction."""
from pathlib import Path
from typing import Optional

import click
import pandas as pd

from emmo.constants import MODEL_DIRECTORY
from emmo.models.prediction import PredictorMHC2


@click.command()
@click.option(
    "--input_file",
    "-i",
    type=Path,
    required=True,
    help="Path to the input csv file.",
)
@click.option(
    "--output_file",
    "-o",
    type=Path,
    required=True,
    help="Path to the output csv file.",
)
@click.option(
    "--model",
    "-m",
    type=str,
    required=True,
    help="Path to the model directory.",
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
@click.option(
    "--peptide_column",
    type=str,
    required=False,
    default="peptide",
    help="The name of column in the csv file containing the peptide.",
)
@click.option(
    "--allele_alpha_column",
    type=str,
    required=False,
    default="allele_alpha",
    help="The name of column in the csv file containing the alpha chain.",
)
@click.option(
    "--allele_beta_column",
    type=str,
    required=False,
    default="allele_beta",
    help="The name of column in the csv file containing the beta chain.",
)
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
    input_file: Path,
    output_file: Path,
    model: str,
    model_name: Optional[str],
    peptide_column: str,
    allele_alpha_column: str,
    allele_beta_column: str,
    force: bool,
    disable_offset_weight: bool,
    length_scoring: bool,
) -> None:
    """Run the prediction for MHC2 peptides and alleles."""
    if output_file.exists():
        if not force:
            raise FileExistsError(
                f"the output file {output_file} already exists, use --force to overwrite"
            )
        elif output_file.is_dir():
            raise ValueError(f"the output path {output_file} exists and is a directory")

    model_path, directory_name = _get_model_path_and_name(model)
    predictor = PredictorMHC2.load(model_path)

    model_name = model_name if model_name is not None else directory_name

    df = pd.read_csv(input_file)

    # run the prediction
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

    df.to_csv(output_file, index=False)


def _get_model_path_and_name(model: str) -> tuple[Path, str]:
    """Return the model path and the name of the model directory.

    Args:
        model: The model path or name of the precompiled models in the models folder of the
            repository.

    Returns:
        The (possibly inferred) path to the model and the name of the model directory.
    """
    model_path = Path(model)

    if not model_path.exists():
        # if path does not exist, try to find the precompiled model this will only work, if the
        # models folder has the correct location relative to the package
        model_path = MODEL_DIRECTORY / "binding-predictor" / model

    if not model_path.exists():
        raise ValueError(
            f"Was not able to identify location of model {model}. "
            "Make sure that the `models` folder is reachable relative to "
            "the package location or provide or valid path (absolute or "
            "relative to working directory)."
        )
    elif not model_path.is_dir():
        raise ValueError(f"The model path {model_path} exists but is not a directory.")

    return model_path, model_path.name
