"""Command line tools for deconvolution."""
from __future__ import annotations

import functools
from pathlib import Path
from typing import Any
from typing import Callable

import click
from cloudpathlib import AnyPath
from cloudpathlib import CloudPath

from emmo.pipeline.sequences import SequenceManager
from emmo.utils import logger
from emmo.utils.click import abort_if_not_exists

log = logger.get(__name__)

REQUIRED_COLUMNS_PER_ALLELE_DECONV = ("peptide", "allele_alpha", "allele_beta")


def common_parameters(func: Callable) -> Callable:
    """Decorator used to define common parameters for full and per-allele deconvolution."""

    @click.option(
        "--motif_length",
        "-m",
        type=int,
        required=False,
        default=9,
        help="The length of the motifs/classes to be discovered.",
    )
    @click.option(
        "--min_classes",
        "-s",
        type=int,
        required=False,
        default=1,
        help="The smallest number of motifs/classes for which to run the EM algorithm.",
    )
    @click.option(
        "--max_classes",
        "-l",
        type=int,
        required=True,
        help="The largest number of motifs/classes for which to run the EM algorithm.",
    )
    @click.option(
        "--number_of_runs",
        "-n",
        type=int,
        required=False,
        default=20,
        help=(
            "The number of EM runs (different initializations) per number of classes, and from "
            "which to choose the best run based on the highest log likelihood."
        ),
    )
    @click.option(
        "--disable_tf",
        is_flag=True,
        help=(
            "If this flag is added, the use of Tensorflow will be disabled and the Python version "
            "of the algorithm will be used instead."
        ),
    )
    @click.option(
        "--output_all_runs",
        is_flag=True,
        help=(
            "If this flag is added, then the results (PPMs, weights, and responsibilities) of all "
            "runs are written into the output directory."
        ),
    )
    @click.option(
        "--force",
        is_flag=True,
        help=(
            "If this flag is added, the deconvolution is run and output is written even if the "
            "output directory already exists."
        ),
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


@click.command()
@click.option(
    "--input_file",
    "-i",
    type=AnyPath,
    required=True,
    callback=abort_if_not_exists,
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
@common_parameters
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
    force: bool,
) -> None:
    """Run the deconvolution for MHC2 ligands."""
    if output_directory.is_dir():
        if not force:
            raise FileExistsError(
                f"the output directory {output_directory} already exists, "
                f"use --force to overwrite"
            )
    elif output_directory.exists():
        raise FileExistsError(f"path {output_directory} exists but is not a directory")

    sequence_manager = SequenceManager.load_from_txt(input_file)

    _run_deconvolutions_per_peptide_list(
        sequence_manager,
        output_directory,
        background_frequencies,
        motif_length,
        min_classes,
        max_classes,
        number_of_runs,
        disable_tf,
        output_all_runs,
        force,
    )


def _run_deconvolutions_per_peptide_list(
    sequence_manager: SequenceManager,
    output_directory: Path | CloudPath,
    background_frequencies: str,
    motif_length: int,
    min_classes: int,
    max_classes: int,
    number_of_runs: int,
    disable_tf: bool,
    output_all_runs: bool,
    force: bool,
) -> None:
    runner_class: Any
    if not disable_tf:
        from emmo.em.mhc2_tf import EMRunnerMHC2 as EMRunnerMHC2_tf

        runner_class = EMRunnerMHC2_tf
    else:
        from emmo.em.mhc2 import EMRunnerMHC2

        runner_class = EMRunnerMHC2

    for i in range(min_classes, max_classes + 1):
        output_dir_i = output_directory / f"classes_{i}"

        em_runner = runner_class(sequence_manager, motif_length, i, background_frequencies)
        em_runner.run(
            output_dir_i, n_runs=number_of_runs, output_all_runs=output_all_runs, force=force
        )
