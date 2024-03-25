"""Command line tools for deconvolution."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import click
import pandas as pd
from cloudpathlib import AnyPath
from cloudpathlib import CloudPath

from emmo.io.file import load_csv
from emmo.pipeline.background import Background
from emmo.pipeline.background import BackgroundType
from emmo.pipeline.sequences import SequenceManager
from emmo.utils import logger
from emmo.utils.alleles import parse_allele_pair
from emmo.utils.click import arguments
from emmo.utils.click import callback
from emmo.utils.viz import plot_motifs_and_length_distribution_per_allele_mhc2

log = logger.get(__name__)


@click.command()
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
    default="MHC2_biondeep",
    help="The background frequencies to be used.",
)
@arguments.deconvolution
def deconvolute_mhc2(
    input_file: Path | CloudPath,
    output_directory: Path | CloudPath,
    background_frequencies: str,
    motif_length: int,
    min_classes: int,
    max_classes: int,
    number_of_runs: int,
    disable_tf: bool,
    output_all_runs: bool,
    skip_existing: bool,
    force: bool,
) -> None:
    """Run the deconvolution for MHC2 ligands."""
    _check_output_directory(output_directory, force, skip_existing)

    sequence_manager = SequenceManager.load_from_txt(input_file)

    _run_deconvolutions_mhc2(
        sequence_manager,
        output_directory,
        background_frequencies,
        motif_length,
        min_classes,
        max_classes,
        number_of_runs,
        disable_tf,
        output_all_runs,
        skip_existing,
    )


@click.command()
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
    "--use_existing_background",
    "-b",
    type=str,
    required=False,
    default=None,
    help="The background frequencies to be used.",
)
@arguments.peptide_and_allele_columns_mhc2
@arguments.deconvolution
@click.option(
    "--plot",
    is_flag=True,
    help="If this flag is set, the deconvolution results are plotted in the subdirectory 'plots'.",
)
@click.pass_context
def deconvolute_per_allele_mhc2(
    ctx: click.core.Context,
    input_file: Path | CloudPath,
    output_directory: Path | CloudPath,
    use_existing_background: str | None,
    peptide_column: str,
    allele_alpha_column: str,
    allele_beta_column: str,
    motif_length: int,
    min_classes: int,
    max_classes: int,
    number_of_runs: int,
    disable_tf: bool,
    output_all_runs: bool,
    force: bool,
    skip_existing: bool,
    plot: bool,
) -> None:
    """Run the per-allele (alpha-beta pair) deconvolution for MHC2 ligands."""
    _check_output_directory(output_directory, force, skip_existing)

    df = load_csv(input_file, usecols=[peptide_column, allele_alpha_column, allele_beta_column])

    background = None
    background_file_path = output_directory / "background.json"
    existing_background = (
        Background.load(background_file_path) if background_file_path.exists() else None
    )

    if use_existing_background is not None:
        if (
            not force
            and existing_background is not None
            and existing_background.name != use_existing_background
        ):
            raise ValueError(
                "background names do not match: "
                f"'use_existing_background={use_existing_background}' vs "
                f"'{existing_background.name}' in existing file {background_file_path}, "
                "use --force to overwrite"
            )

        background = Background(use_existing_background)
        background.save(background_file_path, force=True)
        log.info(f"Using existing background frequencies '{background.name}'")
    elif skip_existing and existing_background is not None:
        background = existing_background
        log.info(f"Using background frequencies loaded from file {background_file_path}")
    elif skip_existing and existing_background is None:
        log.warning(f"Flag --skip_existing is set but {background_file_path} does not exist")

    # background has not been set so far
    if background is None:
        background = Background.from_sequences(df[peptide_column])
        background.save(background_file_path, force=True)
        log.info("Using background frequencies calculated from input peptides")

    for (allele_alpha, allele_beta), df_allele in df.groupby(
        by=[allele_alpha_column, allele_beta_column]
    ):
        allele = f"{allele_alpha}-{allele_beta}"
        sequence_manager = SequenceManager(df_allele[peptide_column].to_list())
        output_directory_allele = output_directory / allele

        log.info(f"Starting to process allele {allele} ...")

        _run_deconvolutions_mhc2(
            sequence_manager,
            output_directory_allele,
            background,
            motif_length,
            min_classes,
            max_classes,
            number_of_runs,
            disable_tf,
            output_all_runs,
            skip_existing,
        )

    if plot:
        ctx.invoke(
            plot_deconvolution_per_allele_mhc2,
            input_directory=output_directory,
            output_directory=(output_directory / "plots"),
            force=True,
        )


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
def plot_deconvolution_per_allele_mhc2(
    input_directory: Path | CloudPath,
    output_directory: Path | CloudPath | None,
    force: bool,
) -> None:
    """Plot per-allele (alpha-beta pair) deconvolution results for MHC2 ligands."""
    if output_directory is None:
        output_directory = input_directory / "plots"

    _check_output_directory(output_directory, force, skip_existing=False)

    data = []

    for allele_dir in input_directory.iterdir():
        if not allele_dir.is_dir():
            continue

        try:
            allele_alpha, allele_beta = parse_allele_pair(allele_dir.name)
        except ValueError:
            log.warning(
                f"input directory contains the subdirectory '{allele_dir.name}' from which the "
                "alpha and beta alleles could not be parsed"
            )
            continue

        for model_dir in allele_dir.iterdir():
            if not model_dir.is_dir():
                continue

            # backward compatibility: the current naming convention is 'classes_{num_of_classes}',
            # the old naming convention is 'clusters{num_of_classes}'
            match = re.match(r"((classes_)|(clusters))([1-9]\d*)", model_dir.name)
            if not match:
                log.warning(
                    f"directory '{allele_dir}' contains a subdirectory '{model_dir.name}' from "
                    "which the number of classes could not be parsed, it will be skipped"
                )
                continue

            number_of_classes = int(match.group(4))

            data.append(
                {
                    "allele_alpha": allele_alpha,
                    "allele_beta": allele_beta,
                    "number_of_classes": number_of_classes,
                    "model_path": model_dir,
                }
            )

    if not data:
        raise ValueError(f"no model directories found in input directory '{input_directory}'")

    df_model_dirs = pd.DataFrame(data).sort_values(
        by=["allele_alpha", "allele_beta", "number_of_classes"], ignore_index=True
    )

    plot_motifs_and_length_distribution_per_allele_mhc2(df_model_dirs, output_directory)


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


def _run_deconvolutions_mhc2(
    sequence_manager: SequenceManager,
    output_directory: Path | CloudPath,
    background_frequencies: BackgroundType,
    motif_length: int,
    min_classes: int,
    max_classes: int,
    number_of_runs: int,
    disable_tf: bool,
    output_all_runs: bool,
    skip_existing: bool,
) -> None:
    """Run deconvolution for the given sequences.

    Args:
        sequence_manager: The SequenceManager instances containing the sequences.
        output_directory: The output directory.
        background_frequencies: The background frequencies to be used.
        motif_length: The length of the motif(s) to be discovered.
        min_classes: The minimum number of classes for which to run the deconvolution.
        max_classes: The maximum number of classes for which to run the deconvolution.
        number_of_runs: The number of EM runs per number of classes.
        disable_tf: Whether to disable the TensorFlow-based implementation of the EM algorithm.
        output_all_runs: Whether to output all EM runs and not just the best per number of classes.
        skip_existing: Whether to skip the deconvolution for existing models.
    """
    runner_class: Any
    if not disable_tf:
        from emmo.em.mhc2_tf import EMRunnerMHC2 as EMRunnerMHC2_tf

        runner_class = EMRunnerMHC2_tf
    else:
        from emmo.em.mhc2 import EMRunnerMHC2

        runner_class = EMRunnerMHC2

    for i in range(min_classes, max_classes + 1):
        output_dir_i = output_directory / f"classes_{i}"

        if skip_existing and (output_dir_i / "model_specs.json").exists():
            log.info(
                f"Skipping deconvolution for {i} class{'es' if i != 1 else ''}, "
                f"found existing model in {output_dir_i}"
            )
            continue

        em_runner = runner_class(sequence_manager, motif_length, i, background_frequencies)
        em_runner.run(
            output_dir_i, n_runs=number_of_runs, output_all_runs=output_all_runs, force=True
        )
