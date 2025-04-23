"""Module for functions to plot deconvolution results like motifs and length distributions."""
from __future__ import annotations

import concurrent.futures
import math
from typing import Any

import logomaker as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure

from emmo.constants import NATURAL_AAS
from emmo.io.file import load_csv
from emmo.io.file import Openable
from emmo.models.cleavage import CleavageModel
from emmo.models.deconvolution import DeconvolutionModel
from emmo.models.deconvolution import DeconvolutionModelMHC1
from emmo.models.deconvolution import DeconvolutionModelMHC2
from emmo.models.prediction import PredictorMHC2
from emmo.pipeline.background import Background
from emmo.pipeline.background import BackgroundType
from emmo.utils import logger
from emmo.utils.alleles import parse_mhc1_allele_pair
from emmo.utils.alleles import parse_mhc2_allele_pair
from emmo.utils.viz.common import save_plot

log = logger.get(__name__)


LOGO_PROPS = {"fade_below": 0.5, "shade_below": 0.5}
COLOR_DEFAULT = "royalblue"
COLOR_DEFAULT_LIGHT = "skyblue"
COLOR_ALT = "sienna"
COLOR_ALT_LIGHT = "bisque"


def plot_single_ppm(
    ppm: np.ndarray,
    alphabet: str | list[str] | tuple[str, ...] | None = None,
    background: BackgroundType | None = None,
    ax: Axes | None = None,
) -> None:
    """Plot a position probability matrix.

    Args:
        ppm: Position probability matrix.
        alphabet: The alphabet. If this is not provided, the natural amino acids are used.
        background: Whether and which background frequencies to use.
        ax: If provided, this Axes instance is used for plotting.
    """
    if ax is None:
        _, ax = plt.subplots()

    if alphabet is None:
        alphabet = NATURAL_AAS

    _df = pd.DataFrame(ppm, columns=list(alphabet))

    # make position 1-based
    _df.index += 1

    if background is not None:
        _df = lm.transform_matrix(
            _df,
            background=Background(background).frequencies,
            from_type="probability",
            to_type="information",
        )

    lm.Logo(_df, **LOGO_PROPS, ax=ax)


def plot_deconvolution_model_motifs(
    model: DeconvolutionModel,
    title: str = "",
    background: BackgroundType | None = None,
    axs: Axes | np.ndarray[Axes] | None = None,
) -> None:
    """Plot the motifs of a deconvolution model.

    Args:
        model: The model.
        title: The title for the plot.
        background: The background frequencies to be used. If this is not provided and the model
            is of type DeconvolutionModelMHC2, then the background associated with the model will be
            used.
        axs: If provided, the Axes instance(s) used for plotting.
    """
    if axs is None:
        number_of_classes = model.number_of_classes
        _, axs = plt.subplots(1, number_of_classes, figsize=(6 * number_of_classes, 4))

    _plot_model_to_axes(model, title, axs, background=background)
    plt.tight_layout()
    plt.show()


def plot_mhc2_model(
    model: DeconvolutionModelMHC2,
    title: str = "",
    axs: Axes | np.ndarray[Axes] | None = None,
) -> None:
    """Plot an MHC2 deconvolution model.

    Args:
        model: The model.
        title: The title for the plot.
        axs: If provided, the Axes instance(s) used for plotting.
    """
    plot_deconvolution_model_motifs(model, title, axs=axs)


def plot_cleavage_model(model: CleavageModel, save_as: Openable | None = None) -> None:
    """Plot a cleavage model.

    Args:
        model: The cleavage model.
        save_as: If provided, save the plot at this location.
    """
    number_of_classes = model.number_of_classes

    cum_class_weights_n = model.class_weights_n
    cum_class_weights_c = model.class_weights_c

    _, axs = plt.subplots(2, number_of_classes, figsize=(4 * number_of_classes, 8))

    if number_of_classes == 1:
        axs = axs[:, np.newaxis]

    for i, name, ppm, cum_class_weights in zip(
        range(2),
        ["N-terminus", "C-terminus"],
        [model.ppm_n, model.ppm_c],
        [cum_class_weights_n, cum_class_weights_c],
    ):
        for j in range(number_of_classes):
            ax = axs[i, j]

            plot_single_ppm(
                ppm[j],
                alphabet=tuple(model.alphabet),
                background=model.background,
                ax=ax,
            )

            ax.set_title(
                f"{name}\nmotif {j+1} of {number_of_classes} (weight {cum_class_weights[j]:.3f})"
            )

    _unify_ylim(axs)
    plt.tight_layout()

    if save_as is not None:
        save_plot(save_as)
    else:
        plt.show()


def plot_motifs_and_length_distribution_per_group(
    mhc_class: int,
    df_model_dirs: pd.DataFrame,
    output_directory: Openable,
    background: BackgroundType | None = None,
    max_groups_per_page: int = 20,
    number_of_processes: int = 1,
) -> None:
    """Plot the motifs of the MHC deconvolution models and the length distributions.

    For each gene (A, B, C, DR, DP, DQ etc.; if existent) and each available number of classes, one
    plot is created that contains the motifs for each class as well as the length distribution of
    the peptides that where assigned to the respective motif during the deconvolution (maximum
    responsibility). If the groups cannot be parsed as alleles (or alpha/beta allele combination in
    case of MHC2), the plots are only split by the number of classes. If there are more than
    'max_groups_per_page' in a plot, the generated PDF will contain multiple pages.

    Args:
        mhc_class: The MHC class.
        df_model_dirs: DataFrame containing the following columns:
            - group: Group; MHC1 allele; or MHC2 alpha and beta allele separated by a hyphen.
            - number_of_classes: Number of classes.
            - model_path: Path to the fitted deconvolution model.
        output_directory: Local or remote directory where to save the plots.
        background: Background frequencies to use. If None, the uniprot background is used.
        max_groups_per_page: Maximum number of groups per page.
        number_of_processes: Number of processes to use for parallelization.
    """
    output_directory = AnyPath(output_directory)
    tasks = _plot_motifs_and_length_distribution_per_group_get_task(
        mhc_class, df_model_dirs, background, max_groups_per_page
    )

    if number_of_processes > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=number_of_processes) as executor:
            futures = [
                executor.submit(_plot_motifs_and_length_distribution_per_group_single_page, *task)
                for task in tasks
            ]
            concurrent.futures.wait(futures)
            figures = [future.result() for future in futures]
    else:
        figures = [
            _plot_motifs_and_length_distribution_per_group_single_page(*task) for task in tasks
        ]

    file_name2figures: dict[str, list[Figure]] = {}
    for (_, _, gene, number_of_classes, _, _), fig in zip(tasks, figures):
        file_name = f"{gene}_classes_{number_of_classes}.pdf"

        if file_name not in file_name2figures:
            file_name2figures[file_name] = []

        file_name2figures[file_name].append(fig)

    if number_of_processes > 1:
        with concurrent.futures.ProcessPoolExecutor(max_workers=number_of_processes) as executor:
            futures = [
                executor.submit(save_plot, output_directory / file_name, figures)
                for file_name, figures in file_name2figures.items()
            ]
            concurrent.futures.wait(futures)
    else:
        for file_name, figures in file_name2figures.items():
            save_plot(output_directory / file_name, figures)


def _plot_motifs_and_length_distribution_per_group_get_task(
    mhc_class: int,
    df_model_dirs: pd.DataFrame,
    background: BackgroundType | None,
    max_groups_per_page: int,
) -> list[tuple[Any, ...]]:
    """Get the tasks for plotting the motifs and length distributions per group.

    Args:
        mhc_class: The MHC class.
        df_model_dirs: DataFrame containing the following columns:
            - group: Group; MHC1 allele; or MHC2 alpha and beta allele separated by a hyphen.
            - number_of_classes: Number of classes.
            - model_path: Path to the fitted deconvolution model.
        background: Background frequencies to use.
        max_groups_per_page: Maximum number of groups per page.

    Returns:
        List of tasks.
    """
    df_model_dirs = df_model_dirs.copy()

    try:
        # if the allele is a valid MHC allele, extract the gene
        if mhc_class == 1:
            df_model_dirs["gene"] = df_model_dirs["group"].apply(
                lambda x: parse_mhc1_allele_pair(x)[0]
            )
        else:
            df_model_dirs["gene"] = df_model_dirs["group"].apply(
                lambda x: parse_mhc2_allele_pair(x)[0][:2]
            )
        log.info("Extracted gene from group column")
    except ValueError:
        df_model_dirs["gene"] = f"MHC{mhc_class}"
        log.info("Could not extract gene from group column, not splitting plots")

    tasks: list[tuple[Any, ...]] = []

    for (gene, number_of_classes), df_gene in df_model_dirs.groupby(["gene", "number_of_classes"]):
        log.info(
            f"Plotting deconvolution results: {gene}, "
            f"{number_of_classes} class{'es' if number_of_classes != 1 else ''}"
        )

        num_pages = math.ceil(len(df_gene) / max_groups_per_page)

        for page in range(num_pages):
            df_gene_page = df_gene.iloc[
                page * max_groups_per_page : (page + 1) * max_groups_per_page
            ]
            tasks.append(
                (
                    mhc_class,
                    df_gene_page,
                    gene,
                    number_of_classes,
                    page,
                    background,
                )
            )

    return tasks


def _plot_motifs_and_length_distribution_per_group_single_page(
    mhc_class: int,
    df_: pd.DataFrame,
    gene: str,
    number_of_classes: int,
    page: int,
    background: BackgroundType | None = None,
) -> Figure:
    """Plot the motifs of the MHC deconvolution models and the length distributions.

    Args:
        mh_class: The MHC class.
        df_: DataFrame containing the following columns:
            - group: Group; MHC1 allele; or MHC2 alpha and beta allele separated by a hyphen.
            - model_path: Path to the fitted deconvolution model.
        gene: The gene.
        number_of_classes: Number of classes.
        page: The page number.
        background: Background frequencies to use. If None, the uniprot background (MHC1) or the
            background of the model (MHC2) is used.

    Raises:
        ValueError: If the number of classes in the model does not match the expected number of
            classes.

    Returns:
        The Figure instance.
    """
    num_of_groups = len(df_)

    n_columns = 3 * number_of_classes + 2
    fig, axs = plt.subplots(num_of_groups, n_columns, figsize=(6 * n_columns, 4 * num_of_groups))

    if num_of_groups == 1:
        axs = axs[np.newaxis, :]

    for i, (_, row) in enumerate(df_.iterrows()):
        group = str(row.group)
        log.info(f"Plotting gene {gene}, page {page}, group {i+1}/{num_of_groups} ({group})")

        model_path = AnyPath(row.model_path)
        model: DeconvolutionModel

        if mhc_class == 1:
            model = DeconvolutionModelMHC1.load(model_path)
            if background is None:
                background = Background("uniprot")
                log.info(
                    "No background frequencies provided for plotting MHC1 models; using default "
                    "(uniprot)"
                )
        else:
            model = DeconvolutionModelMHC2.load(model_path)
            if background is None:
                background = model.background

        if model.number_of_classes != number_of_classes:
            raise ValueError(
                f"number of classes in model {row.model_path} does not match, should be "
                f"{number_of_classes}, got {model.number_of_classes}"
            )

        _plot_model_to_axes(model, group, axs[i, :-2:3], background)
        _plot_class_weights_to_axes(model, group, list(axs[i, 1:-2:3]) + [axs[i, -2]])
        _plot_length_distribution_from_responsibilities(
            model,
            load_csv(model_path / "responsibilities.csv"),
            group,
            list(axs[i, 2:-2:3]) + [axs[i, -1]],
            include_flat=True,
        )

    _unify_ylim(axs[:, :-2:3])
    fig.tight_layout()

    return fig


def plot_predictor_mhc2(predictor: PredictorMHC2, output_directory: Openable) -> None:
    """Plot the summary of an MHC2 predictor (motifs of alleles and cleavage model).

    Args:
        predictor: The MHC2 binding predictor.
        output_directory: Local or remote directory where to save the plots.
    """
    output_directory = AnyPath(output_directory)

    plot_cleavage_model(predictor.cleavage_model, output_directory / "cleavage_model.pdf")

    genes = sorted({allele[:2] for allele in predictor.ppms})

    for gene in genes:
        log.info(f"Plotting motifs for {gene} ...")
        alleles = sorted([allele for allele in predictor.ppms if allele.startswith(gene)])

        num_of_alleles = len(alleles)
        _, axs = plt.subplots(num_of_alleles, 1, figsize=(6, 4 * num_of_alleles))

        for i, allele in enumerate(alleles):
            log.info(f"Plotting {gene} allele {i+1}/{num_of_alleles} ({allele})")
            ax = axs[i]
            plot_single_ppm(
                predictor.ppms[allele],
                alphabet=predictor.alphabet,
                background=predictor.background,
                ax=ax,
            )
            title = allele

            if "selected_motifs" in predictor.compilation_details:
                details = predictor.compilation_details["selected_motifs"][allele]
                title = (
                    f"{title}\n"
                    f"motif {details['motif']} of {details['classes']} "
                    f"(weight {details['motif_weight']:.3f})"
                )
            ax.set_title(title)

        _unify_ylim(axs)
        plt.tight_layout()

        file_path = output_directory / f"motifs_{gene}.pdf"
        save_plot(file_path)


def _unify_ylim(axs: np.ndarray[Axes]) -> None:
    """Unify the the ylim of Axes objects to the minimum and maximum.

    Args:
        axs: Array of Axes instances.
    """
    axs = axs.flatten()
    min_ylim = min(ax.get_ylim()[0] for ax in axs)
    max_ylim = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(min_ylim, max_ylim)


def _get_weights_per_motif(model: DeconvolutionModel) -> np.ndarray | None:
    """Get the weights per motif.

    Args:
        model: The model.

    Returns:
        The weights per motif.

    Raises:
        ValueError: If the model type is not supported.
    """
    if isinstance(model, DeconvolutionModelMHC1):
        return model.class_weights[9] if 9 in model.class_weights else None
    elif isinstance(model, DeconvolutionModelMHC2):
        return np.sum(model.class_weights, axis=1)
    else:
        raise ValueError(f"unsupported model type: {type(model)}")


def _plot_model_to_axes(
    model: DeconvolutionModel,
    title: str,
    axs: Axes | np.ndarray,
    background: BackgroundType | None = None,
) -> None:
    """Plot an deconvolution model using the given Axes instance(s).

    Args:
        model: The model.
        title: The title for the plot.
        axs: The Axes instance(s) used for plotting.
        background: The background frequencies to use.
    """
    class_weights = _get_weights_per_motif(model)
    number_of_classes = model.number_of_classes

    if background is None and isinstance(model, DeconvolutionModelMHC2):
        background = model.background

    for i in range(number_of_classes):
        ax = axs if isinstance(axs, Axes) and i == 0 else axs[i]

        plot_single_ppm(
            model.ppm[i],
            alphabet=tuple(model.alphabet),
            background=background,
            ax=ax,
        )

        if class_weights is not None:
            _title = f"motif (class {i+1} of {number_of_classes}, weight {class_weights[i]:.3f})"
        else:
            _title = f"motif (class {i+1} of {number_of_classes})"

        if title:
            _title = f"{title}\n{_title}"
        ax.set_title(_title)


def _plot_class_weights_to_axes(
    model: DeconvolutionModel,
    title: str,
    axs: Axes | np.ndarray,
    include_flat: bool = True,
) -> None:
    """Plot the class weights of an MHC deconvolution model using the given Axes instance(s).

    Args:
        model: The model.
        title: The title for the plot.
        axs: The Axes instance(s) used for plotting.
        include_flat: Whether to include the flat motif.

    Raises:
        ValueError: If the model type is not supported.
    """
    if isinstance(model, DeconvolutionModelMHC1):
        _plot_mhc1_class_weights_to_axes(model, title, axs, include_flat=include_flat)
    elif isinstance(model, DeconvolutionModelMHC2):
        _plot_mhc2_offset_weights_to_axes(model, title, axs, include_flat=include_flat)
    else:
        raise ValueError(f"unsupported model type: {type(model)}")


def _plot_mhc1_class_weights_to_axes(
    model: DeconvolutionModelMHC1,
    title: str,
    axs: Axes | np.ndarray,
    include_flat: bool = True,
) -> None:
    """Plot the class weights of an MHC1 deconvolution model using the given Axes instance(s).

    Args:
        model: The model.
        title: The title for the plot.
        axs: The Axes instance(s) used for plotting.
        include_flat: Whether to include the flat motif.

    Raises:
        RuntimeError: If the provided Axes array is not large enough.
    """
    if isinstance(axs, Axes):
        axs = [axs]

    number_of_classes = model.number_of_classes
    n_axes = number_of_classes + 1 if include_flat else number_of_classes

    if n_axes != len(axs):
        raise ValueError(f"invalid number of Axes: {n_axes} needed, {len(axs)} provided")

    lengths = sorted(model.class_weights.keys())

    for i in range(n_axes):
        ax = axs[i]

        if i < number_of_classes:
            class_label = str(i + 1)
            color_full_values = COLOR_DEFAULT_LIGHT
            color_partial_values = COLOR_DEFAULT
            _title = f"class {class_label} of {number_of_classes}"
        else:
            class_label = "flat"
            color_full_values = COLOR_ALT_LIGHT
            color_partial_values = COLOR_ALT
            _title = "flat motif"

        weights = [model.class_weights[length][i] for length in lengths]
        ax.bar(lengths, [1 for _ in lengths], color=color_full_values)
        ax.bar(lengths, weights, color=color_partial_values)
        ax.set_xticks(lengths)
        ax.set_xticklabels(ax.get_xticks(), fontsize=8, rotation=90)
        ax.set_ylim(0, 1.1)

        # annotate the bars with the weights
        for length, weight in zip(lengths, weights):
            ax.text(length, 1.05, f"{weight:.3f}", ha="center", va="center", fontsize=8)

        _title = f"per-length weights ({_title})"
        if title:
            _title = f"{title}\n{_title}"
        ax.set_title(_title)


def _plot_mhc2_offset_weights_to_axes(
    model: DeconvolutionModelMHC2,
    title: str,
    axs: Axes | np.ndarray,
    include_flat: bool = True,
) -> None:
    """Plot the offset weights of an MHC2 deconvolution model using the given Axes instance(s).

    Args:
        model: The model.
        title: The title for the plot.
        axs: The Axes instance(s) used for plotting.
        include_flat: Whether to include the flat motif.

    Raises:
        RuntimeError: If the provided Axes array is not large enough.
    """
    if isinstance(axs, Axes):
        axs = [axs]

    number_of_classes = model.number_of_classes
    n_axes = number_of_classes + 1 if include_flat else number_of_classes

    if n_axes != len(axs):
        raise ValueError(f"invalid number of Axes: {n_axes} needed, {len(axs)} provided")

    cum_class_weights = np.sum(model.class_weights, axis=1)
    cum_offset_weights = np.sum(model.class_weights, axis=0)

    for i in range(n_axes):
        ax = axs[i]

        if i < number_of_classes:
            class_label = str(i + 1)
            color_full_values = COLOR_DEFAULT_LIGHT
            color_partial_values = COLOR_DEFAULT
            _title = f"class {class_label} of {number_of_classes}"
        else:
            class_label = "flat"
            color_full_values = COLOR_ALT_LIGHT
            color_partial_values = COLOR_ALT
            _title = "flat motif"

        offsets = model.aligned_offsets.get_offset_annotation()
        ax.bar(offsets, cum_offset_weights, color=color_full_values)
        ax.bar(offsets, model.class_weights[i], color=color_partial_values)
        ax.set_xticks(offsets)
        ax.set_xticklabels(ax.get_xticks(), fontsize=8, rotation=90)

        _title = f"offset weights ({_title}, total weight {cum_class_weights[i]:.3f})"
        if title:
            _title = f"{title}\n{_title}"
        ax.set_title(_title)


def _plot_length_distribution_from_responsibilities(
    model: DeconvolutionModel,
    responsibilities: pd.DataFrame,
    title: str,
    axs: Axes | np.ndarray[Axes] | list[Axes],
    include_flat: bool = True,
) -> None:
    """Plot the length distribution according to the best responsibility values.

    Args:
        model: The deconvolution model.
        responsibilities: The responsibilities dataframe.
        title: The title for the plot.
        axs: The Axes instance(s) used for plotting.
        include_flat: Whether to include the flat motif.
    """
    if isinstance(axs, Axes):
        axs = [axs]

    number_of_classes = model.number_of_classes
    n_axes = number_of_classes + 1 if include_flat else number_of_classes

    if n_axes != len(axs):
        raise ValueError(f"invalid number of Axes: {n_axes} needed, {len(axs)} provided")

    responsibilities = responsibilities.assign(length=responsibilities["peptide"].str.len())
    class_weights = _get_weights_per_motif(model)

    for i in range(n_axes):
        ax = axs[i]

        if i < number_of_classes:
            class_label = str(i + 1)
            color_full_values = COLOR_DEFAULT_LIGHT
            color_partial_values = COLOR_DEFAULT
            _title = f"class {class_label} of {number_of_classes}"
        else:
            class_label = "flat"
            color_full_values = COLOR_ALT_LIGHT
            color_partial_values = COLOR_ALT
            _title = "flat motif"

        is_assigned = responsibilities["best_class"].astype(str) == class_label
        _plot_length_distribution(
            responsibilities["length"],
            length_values_partial=responsibilities[is_assigned]["length"],
            ax=ax,
            color_full_values=color_full_values,
            color_partial_values=color_partial_values,
        )

        if class_weights is not None:
            _title = f"length distribution ({_title}, weight {class_weights[i]:.3f})"
        else:
            _title = f"length distribution ({_title})"

        if title:
            _title = f"{title}\n{_title}"
        ax.set_title(_title)


def _plot_length_distribution(
    length_values: np.ndarray | pd.Series,
    ax: Axes,
    length_values_partial: np.ndarray | pd.Series | None = None,
    color_full_values: str = COLOR_DEFAULT_LIGHT,
    color_partial_values: str = COLOR_DEFAULT,
) -> None:
    """Plot a length distribution.

    Args:
        length_values: The length values.
        ax: The Axes instance used for plotting.
        length_values_partial: Additional length values to plot, these should be a subset of the
            values in 'length_values' for the plot to be meaningful.
        color_full_values: Color used for plotting 'length_values'.
        color_partial_values: Color used for plotting 'length_values_partial'.
    """
    min_length = length_values.min()
    max_length = length_values.max()

    ax.hist(
        length_values,
        bins=[x - 0.5 for x in range(min_length, max_length + 2)],
        rwidth=0.8,
        color=color_full_values,
    )

    if length_values_partial is not None:
        ax.hist(
            length_values_partial,
            bins=[x - 0.5 for x in range(min_length, max_length + 2)],
            rwidth=0.8,
            color=color_partial_values,
        )

    ax.set_xlim(min_length - 0.5, max_length + 0.5)
    ax.set_xticks(list(range(min_length, max_length + 1)))
    ax.set_xticklabels(ax.get_xticks(), fontsize=8, rotation=90)
    ax.tick_params(axis="y", labelsize=8)
