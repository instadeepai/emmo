"""Command line tools for deconvolution."""
from __future__ import annotations
from pathlib import Path
from typing import Any

import click

from emmo.io.sequences import SequenceManager


@click.command()
@click.option(
    "--input_file",
    "-i",
    type=str,
    required=True,
    help="Path to the input file containing the peptides.",
)
@click.option(
    "--output_directory",
    "-o",
    type=str,
    required=True,
    help="Output directory.",
)
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
        "The number of EM runs (different initializations) per number of classes, and from which "
        "to choose the best run based on the highest log likelihood."
    ),
)
@click.option(
    "--disable_tf",
    is_flag=True,
    help=(
        "If this flag is added, the use of Tensorflow will be disabled and the Python version of "
        "the algorithm will be used instead."
    ),
)
@click.option(
    "--output_all_runs",
    is_flag=True,
    help=(
        "If this flag is added, then the results (PPMs, weights, and responsibilities) of all runs "
        "are written into the output directory."
    ),
)
@click.option(
    "--force",
    is_flag=True,
    help=(
        "If this flag is added, the deconvolution is run and output is written even if the output "
        "directory already exists."
    ),
)
def deconvolute_mhc2(
    input_file: str | Path,
    output_directory: str | Path,
    motif_length: int,
    min_classes: int,
    max_classes: int,
    number_of_runs: int,
    disable_tf: bool,
    output_all_runs: bool,
    force: bool,
) -> None:
    """Run the deconvolution for MHC2 ligands."""
    input_file = Path(input_file)
    output_directory = Path(output_directory)

    if output_directory.is_dir():
        if not force:
            raise FileExistsError(
                f"the output directory {output_directory} already exists, "
                f"use --force to overwrite"
            )
    elif output_directory.exists():
        raise FileExistsError(f"path {output_directory} exists but is not a directory")
    else:
        output_directory.mkdir(parents=False, exist_ok=False)

    sm = SequenceManager(input_file)

    runner_class: Any
    if not disable_tf:
        from emmo.em.mhc2_tf import EMRunnerMHC2 as EMRunnerMHC2_tf

        runner_class = EMRunnerMHC2_tf
    else:
        from emmo.em.mhc2 import EMRunnerMHC2

        runner_class = EMRunnerMHC2

    for i in range(min_classes, max_classes + 1):
        output_dir_i = output_directory / f"classes_{i}"
        output_dir_i.mkdir(parents=True, exist_ok=True)

        em_runner = runner_class(sm, motif_length, i)
        em_runner.run(
            output_dir_i,
            n_runs=number_of_runs,
            output_all_runs=output_all_runs,
        )
