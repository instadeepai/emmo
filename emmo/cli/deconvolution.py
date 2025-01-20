"""Command line tools for deconvolution."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import pandas as pd
from cloudpathlib import AnyPath
from cloudpathlib import CloudPath

from emmo.em.mhc1 import EMRunnerMHC1 as EMRunnerMHC1_Python
from emmo.em.mhc1_c import EMRunnerMHC1 as EMRunnerMHC1_C
from emmo.em.mhc2 import EMRunnerMHC2 as EMRunnerMHC2_Python
from emmo.em.mhc2_c import EMRunnerMHC2 as EMRunnerMHC2_C
from emmo.io.file import load_csv
from emmo.io.output import check_valid_directory_name
from emmo.io.output import find_deconvolution_results
from emmo.models.cleavage import CleavageModel
from emmo.pipeline.background import Background
from emmo.pipeline.background import BackgroundType
from emmo.pipeline.sequences import SequenceManager
from emmo.utils import logger
from emmo.utils.click import arguments
from emmo.utils.click import callback
from emmo.utils.viz import plot_cleavage_model
from emmo.utils.viz import plot_motifs_and_length_distribution_per_group_mhc1
from emmo.utils.viz import plot_motifs_and_length_distribution_per_group_mhc2

log = logger.get(__name__)


@click.command()
@arguments.mhc_class
@click.option(
    "--input_file",
    "-i",
    type=AnyPath,
    required=True,
    callback=callback.abort_if_not_exists,
    help="Path to the local or remote input file containing the peptides.",
)
@click.option(
    "--output_directory",
    "-o",
    type=AnyPath,
    required=True,
    help="Local or remote output directory.",
)
@click.option(
    "--background_frequencies",
    "-b",
    type=str,
    required=False,
    default=None,
    help="The background frequencies to be used (relevant for MHC2 only).",
)
@arguments.deconvolution
def deconvolute(
    mhc_class: str,
    input_file: Path | CloudPath,
    output_directory: Path | CloudPath,
    background_frequencies: str,
    motif_length: int,
    min_classes: int,
    max_classes: int,
    number_of_runs: int,
    disable_c_extension: bool,
    output_all_runs: bool,
    skip_existing: bool,
    force: bool,
) -> None:
    """Run the deconvolution for a list of MHC ligands."""
    _check_output_directory(output_directory, force, skip_existing)

    sequence_manager = SequenceManager.load_from_txt(input_file)

    _run_deconvolution(
        mhc_class,
        sequence_manager,
        output_directory,
        background_frequencies,
        motif_length,
        min_classes,
        max_classes,
        number_of_runs,
        disable_c_extension,
        output_all_runs,
        skip_existing,
    )


@click.command()
@arguments.mhc_class
@click.option(
    "--input_file",
    "-i",
    type=AnyPath,
    required=True,
    callback=callback.abort_if_not_exists,
    help="Path to the local or remote input file containing the peptides and alleles.",
)
@click.option(
    "--output_directory",
    "-o",
    type=AnyPath,
    required=True,
    help="Local or remote output directory.",
)
@click.option(
    "--existing_background_name",
    "-b",
    type=str,
    required=False,
    default=None,
    help="The background frequencies to be used (relevant for MHC2 only).",
)
@arguments.peptide_and_group_columns
@arguments.deconvolution
@click.option(
    "--plot",
    is_flag=True,
    help="If this flag is set, the deconvolution results are plotted in the subdirectory 'plots'.",
)
@click.pass_context
def deconvolute_per_group(
    ctx: click.core.Context,
    mhc_class: str,
    input_file: Path | CloudPath,
    output_directory: Path | CloudPath,
    existing_background_name: str | None,
    peptide_column: str,
    group_columns: str | None,
    motif_length: int,
    min_classes: int,
    max_classes: int,
    number_of_runs: int,
    disable_c_extension: bool,
    output_all_runs: bool,
    force: bool,
    skip_existing: bool,
    plot: bool,
) -> None:
    """Run the per-group deconvolution for MHC ligands.

    If the 'group_columns' argument is not provided, the deconvolution is run per allele. In this
    case, a column 'allele' is expected in the input file for MHC1, and columns 'allele_alpha' and
    'allele_beta' are expected for MHC2. If custom columns for the alpha and beta chains are
    provided, these are assumed to be ordered such that the alpha chain comes first.
    """
    _check_output_directory(output_directory, force, skip_existing)
    if group_columns is None:
        group_columns = "allele" if mhc_class == "1" else "allele_alpha,allele_beta"

    group_columns_list = group_columns.split(",")

    df = load_csv(input_file, usecols=[peptide_column, *group_columns_list])

    background = _initialize_background(
        output_directory,
        existing_background_name,
        df[peptide_column],
        force,
        skip_existing,
    )

    for group, df_group in df.groupby(by=group_columns_list):
        group_name = "-".join(group)

        check_valid_directory_name(group_name)
        output_directory_group = output_directory / group_name

        sequence_manager = SequenceManager(df_group[peptide_column].to_list())

        log.info(f"Starting to process {group_name} ...")

        _run_deconvolution(
            mhc_class,
            sequence_manager,
            output_directory_group,
            background,
            motif_length,
            min_classes,
            max_classes,
            number_of_runs,
            disable_c_extension,
            output_all_runs,
            skip_existing,
        )

    if plot:
        ctx.invoke(
            plot_deconvolution_per_group,
            mhc_class=mhc_class,
            input_directory=output_directory,
            output_directory=(output_directory / "plots"),
            force=True,
        )


@click.command()
@arguments.mhc_class
@click.option(
    "--input_directory",
    "-i",
    type=AnyPath,
    required=True,
    callback=callback.abort_if_not_exists,
    help="Path to the local or remote directory containing the deconvolution results.",
)
@click.option(
    "--output_directory",
    "-o",
    type=AnyPath,
    required=False,
    default=None,
    help=(
        "Local or remote output directory. If this is not provided, the plots are saved in a"
        "subdirectory 'plots' within the input directory."
    ),
)
@arguments.force(help="Overwrite existing plots.")
def plot_deconvolution_per_group(
    mhc_class: str,
    input_directory: Path | CloudPath,
    output_directory: Path | CloudPath | None,
    force: bool,
) -> None:
    """Plot per-group deconvolution results for MHC ligands."""
    if output_directory is None:
        output_directory = input_directory / "plots"

    _check_output_directory(output_directory, force, skip_existing=False)
    df_model_dirs = find_deconvolution_results(input_directory)

    if mhc_class == "1":
        backgroud_file_path = input_directory / "background.json"
        if not backgroud_file_path.exists():
            log.warning(
                f"Background file {backgroud_file_path} not found, will use default background"
            )
            background = None
        else:
            background = Background.load(backgroud_file_path)

        plot_motifs_and_length_distribution_per_group_mhc1(
            df_model_dirs, output_directory, background=background
        )
    else:
        plot_motifs_and_length_distribution_per_group_mhc2(df_model_dirs, output_directory)


@click.command()
@click.option(
    "--input_file",
    "-i",
    type=AnyPath,
    required=True,
    callback=callback.abort_if_not_exists,
    help=(
        "Path to the local or remote input file containing the peptides. Can be a CSV file in "
        "which case '--peptide-column <COLUMN_NAME>' is required. Otherwise, it is assumed that "
        "the file contains one peptide per line with no header."
    ),
)
@click.option(
    "--output_directory",
    "-o",
    type=AnyPath,
    required=True,
    help="Local or remote output directory.",
)
@click.option(
    "--existing_background_name",
    "-b",
    type=str,
    required=False,
    default=None,
    help="The background frequencies to be used.",
)
@click.option(
    "--peptide_column",
    type=str,
    required=False,
    default=None,
    help=(
        "If a CSV file is provided as input, the name of the column containing the peptide has to "
        "be provided."
    ),
)
@arguments.deconvolution(motif_length=3)
@click.option(
    "--plot",
    is_flag=True,
    help="If this flag is set, the deconvolution results are plotted in the subdirectory 'plots'.",
)
def deconvolute_for_cleavage_mhc2(
    input_file: Path | CloudPath,
    output_directory: Path | CloudPath,
    existing_background_name: str | None,
    peptide_column: str | None,
    motif_length: int,
    min_classes: int,
    max_classes: int,
    number_of_runs: int,
    disable_c_extension: bool,
    output_all_runs: bool,
    force: bool,
    skip_existing: bool,
    plot: bool,
) -> None:
    """Run the deconvolution for MHC2 cleavage models."""
    _check_output_directory(output_directory, force, skip_existing)

    if peptide_column is not None:
        sequence_manager = SequenceManager.load_from_csv(input_file, peptide_column)
    else:
        sequence_manager = SequenceManager.load_from_txt(input_file)

    if sequence_manager.min_length < motif_length:
        raise ValueError(
            f"minimal sequence length {sequence_manager.min_length} is smaller than "
            f"motif length {motif_length}"
        )

    background = _initialize_background(
        output_directory,
        existing_background_name,
        sequence_manager.sequences,
        force,
        skip_existing,
    )

    for terminus, sequences in zip(
        ("N", "C"),
        (
            [seq[:motif_length] for seq in sequence_manager.sequences],
            [seq[-motif_length:] for seq in sequence_manager.sequences],
        ),
    ):
        output_directory_terminus = output_directory / f"deconvolution-{terminus}-terminus"

        log.info(f"Starting to process {terminus}-terminus ...")

        _run_deconvolution(
            "2",  # MHC class
            SequenceManager(sequences),
            output_directory_terminus,
            background,
            motif_length,
            min_classes,
            max_classes,
            number_of_runs,
            disable_c_extension,
            output_all_runs,
            skip_existing,
        )

        log.info(f"Finished deconvolution for {terminus}-terminus ...")

    for i in range(min_classes, max_classes + 1):
        cleavage_model = CleavageModel.load_from_separate_models(
            output_directory / "deconvolution-N-terminus" / f"classes_{i}",
            output_directory / "deconvolution-C-terminus" / f"classes_{i}",
        )
        output_directory_i = output_directory / f"cleavage_model_classes_{i}"
        cleavage_model.save(output_directory_i, force=True)
        log.info(f"Compiled cleavage model for {i} classes at {output_directory_i}")

        if plot:
            plot_file_path = output_directory / "plots" / f"cleavage_model_classes_{i}.pdf"
            plot_cleavage_model(cleavage_model, save_as=plot_file_path)


def _check_output_directory(
    output_directory: Path | CloudPath, force: bool, skip_existing: bool
) -> None:
    """Check if output directory already exists.

    Args:
        output_directory: Local or remote output directory path
        force: Whether to overwrite files
        skip_existing: Whether to skip the deconvolution for existing models.

    Raises:
        FileExistsError: If the output directory already exists and force is False.
        FileExistsError: If 'output_directory' points to an existing file.
    """
    if force and skip_existing:
        raise ValueError("the flags --force/-f and --skip_existing cannot be used together")

    if output_directory.is_dir():
        if not force and not skip_existing:
            raise FileExistsError(
                f"the output directory {output_directory} already exists, use --force to overwrite"
            )
    elif output_directory.exists():
        raise FileExistsError(f"path {output_directory} exists but is not a directory")


def _run_deconvolution(
    mhc_class: str,
    sequence_manager: SequenceManager,
    output_directory: Path | CloudPath,
    background_frequencies: BackgroundType | None,
    motif_length: int,
    min_classes: int,
    max_classes: int,
    number_of_runs: int,
    disable_c_extension: bool,
    output_all_runs: bool,
    skip_existing: bool,
) -> None:
    """Run MHC deconvolution for the given sequences.

    Args:
        mhc_class: The MHC class for which to run the deconvolution.
        sequence_manager: The SequenceManager instances containing the sequences.
        output_directory: The output directory.
        background_frequencies: The background frequencies to be used. Only for MHC2.
        motif_length: The length of the motif(s) to be discovered.
        min_classes: The minimum number of classes for which to run the deconvolution.
        max_classes: The maximum number of classes for which to run the deconvolution.
        number_of_runs: The number of EM runs per number of classes.
        disable_c_extension: Whether to disable the C-implementation of the EM algorithm.
        output_all_runs: Whether to output all EM runs and not just the best per number of classes.
        skip_existing: Whether to skip the deconvolution for existing models.
    """
    additional_parameters = []
    runner_class: Any

    if mhc_class == "1":
        runner_class = EMRunnerMHC1_Python if disable_c_extension else EMRunnerMHC1_C
    else:
        runner_class = EMRunnerMHC2_Python if disable_c_extension else EMRunnerMHC2_C
        if background_frequencies is not None:
            background_frequencies = "MHC2_biondeep"
            log.info(
                f"Using default background frequencies '{background_frequencies}' for MHC2 "
                "deconvolution"
            )
        additional_parameters.append(background_frequencies)

    for i in range(min_classes, max_classes + 1):
        output_dir_i = output_directory / f"classes_{i}"

        if skip_existing and (output_dir_i / "model_specs.json").exists():
            log.info(
                f"Skipping deconvolution for {i} class{'es' if i != 1 else ''}, "
                f"found existing model in {output_dir_i}"
            )
            continue

        em_runner = runner_class(sequence_manager, motif_length, i, *additional_parameters)
        em_runner.run(
            output_dir_i, n_runs=number_of_runs, output_all_runs=output_all_runs, force=True
        )


def _initialize_background(
    output_directory: Path | CloudPath,
    existing_background_name: str | None,
    sequences: pd.Series | list[str] | None,
    force: bool,
    skip_existing: bool,
) -> Background:
    """Initialize the background distribution to be used in the deconvolution.

    Args:
        output_directory: The output directory where to write the results or where parts of the
            results already exist.
        existing_background_name: The name of a background distribution that is available in the
            package.
        sequences: Sequences from which to calculate a background distribution.
        force: Whether existing deconvolution results shall be overwritten.
        skip_existing: Whether to skip the deconvolution for models that already exist.

    Raises:
        ValueError: If the output directory contains a background distribution and its name and
            'existing_background_name' do not match (and force is False).
        RuntimeError: If the background distribution could not be initialized.

    Returns:
        Background: A background distribution.
    """
    background = None
    background_file_path = output_directory / "background.json"
    existing_background = (
        Background.load(background_file_path) if background_file_path.exists() else None
    )

    if existing_background_name is not None:
        if (
            not force
            and existing_background is not None
            and existing_background.name != existing_background_name
        ):
            raise ValueError(
                "background names do not match: "
                f"'existing_background_name={existing_background_name}' vs "
                f"'{existing_background.name}' in existing file {background_file_path}, "
                "use --force to overwrite"
            )

        background = Background(existing_background_name)
        background.save(background_file_path, force=True)
        log.info(f"Using existing background frequencies '{background.name}'")
    elif skip_existing and existing_background is not None:
        background = existing_background
        log.info(f"Using background frequencies loaded from file {background_file_path}")
    elif skip_existing and existing_background is None:
        log.warning(f"Flag --skip_existing is set but {background_file_path} does not exist")

    # background has not been set so far
    if background is None and sequences is not None:
        background = Background.from_sequences(sequences)
        background.save(background_file_path, force=True)
        log.info("Using background frequencies calculated from input peptides")

    if background is None:
        raise RuntimeError("background distribution could not be initialized")

    return background
