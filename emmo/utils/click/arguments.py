"""Module used to define common arguments used for command lines.

Once defined in this module, an argument (or several arguments) can be
used as follows:

from emmo.utils.click import arguments

....

@arguments.{argument_name}
def my_cli_command(...)
"""
from __future__ import annotations

import functools
from typing import Any
from typing import Callable

import click


def force(func: Callable = None, **kwargs: Any) -> Callable:
    """Decorator used to define the force flag.

    The 'help' parameter for this parameter can be overwritten by calling @force(help="My custom
    help").
    """
    kwargs["help"] = kwargs.get(
        "help",
        "Overwrite any existing columns, datasets or models (both locally or in "
        "the cloud) used by the command. This parameter should be used with caution.",
    )

    def _outer_wrapper(func: Callable) -> Callable:
        @click.option("--force", "-f", is_flag=True, **kwargs)
        @functools.wraps(func)
        def _inner_wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return _inner_wrapper

    if func is None:
        return _outer_wrapper

    return _outer_wrapper(func)


def peptide_and_allele_columns_mhc2(func: Callable) -> Callable:
    """Decorator used to define the column name for peptides and alleles in the input CSV file."""

    @click.option(
        "--peptide_column",
        type=str,
        required=False,
        default="peptide",
        help="The name of column in the CSV file containing the peptide.",
    )
    @click.option(
        "--allele_alpha_column",
        type=str,
        required=False,
        default="allele_alpha",
        help="The name of column in the CSV file containing the alpha chain.",
    )
    @click.option(
        "--allele_beta_column",
        type=str,
        required=False,
        default="allele_beta",
        help="The name of column in the CSV file containing the beta chain.",
    )
    @functools.wraps(func)
    def _wrapper(*args: Any, **kwargs: Any) -> Any:
        return func(*args, **kwargs)

    return _wrapper


def deconvolution(func: Callable = None, **kwargs: Any) -> Callable:
    """Decorator used to define common parameters for full and per-allele deconvolution."""
    # set defaults that are not overridden
    kwargs["motif_length"] = kwargs.get("motif_length", 9)
    kwargs["min_classes"] = kwargs.get("min_classes", 1)

    def _outer_wrapper(func: Callable) -> Callable:
        @click.option(
            "--motif_length",
            "-m",
            type=int,
            required=False,
            default=kwargs["motif_length"],
            help="The length of the motifs/classes to be discovered.",
        )
        @click.option(
            "--min_classes",
            "-s",
            type=int,
            required=False,
            default=kwargs["min_classes"],
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
            required=True,
            help=(
                "The number of EM runs (different initializations) per number of classes, and "
                "from which to choose the best run based on the highest log likelihood."
            ),
        )
        @click.option(
            "--disable_tf",
            is_flag=True,
            help=(
                "If this flag is added, the use of Tensorflow will be disabled and the Python "
                "version of the algorithm will be used instead."
            ),
        )
        @click.option(
            "--output_all_runs",
            is_flag=True,
            help=(
                "If this flag is added, then the results (PPMs, weights, and responsibilities) of "
                "all runs are written into the output directory."
            ),
        )
        @click.option(
            "--skip_existing",
            is_flag=True,
            help=(
                "If this flag is set, the deconvolution is skipped for models that already exist "
                "in the output folder are skipped."
            ),
        )
        @force(
            help=(
                "If this flag is added, the deconvolution is run and output is written even if "
                "the output directory already exists."
            )
        )
        @functools.wraps(func)
        def _inner_wrapper(*args: Any, **kwargs: Any) -> Any:
            return func(*args, **kwargs)

        return _inner_wrapper

    if func is None:
        return _outer_wrapper

    return _outer_wrapper(func)
