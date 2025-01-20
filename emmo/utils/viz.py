"""Module for functions to plot deconvolution results like motifs and length distributions."""
from __future__ import annotations

import tempfile
from pathlib import Path

import logomaker as lm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from cloudpathlib import AnyPath
from cloudpathlib import CloudPath
from matplotlib.axes._axes import Axes

from emmo.bucket.io import upload_to_directory
from emmo.constants import NATURAL_AAS
from emmo.io.file import Openable
from emmo.models.cleavage import CleavageModel
from emmo.models.deconvolution import DeconvolutionModelMHC1
from emmo.models.deconvolution import DeconvolutionModelMHC2
from emmo.models.prediction import PredictorMHC2
from emmo.pipeline.background import Background
from emmo.pipeline.background import BackgroundType
from emmo.utils import logger
from emmo.utils.alleles import parse_mhc1_allele_pair
from emmo.utils.alleles import parse_mhc2_allele_pair

log = logger.get(__name__)


LOGO_PROPS = {"fade_below": 0.5, "shade_below": 0.5}
COLOR_DEFAULT = "royalblue"
COLOR_DEFAULT_LIGHT = "skyblue"
COLOR_ALT = "sienna"
COLOR_ALT_LIGHT = "bisque"


def save_plot(file_path: Openable) -> None:
    """Save the current plot locally or remotely and close it.

    Args:
        file_path: Local or remote path where to save the plot.
    """
    file_path = AnyPath(file_path)

    if isinstance(file_path, CloudPath):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / file_path.name
            plt.savefig(tmp_file)
            upload_to_directory(tmp_file, file_path.parent, force=True)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(file_path)

    log.info(f"Saved plot at {file_path}")

    plt.close()


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
    if axs is None:
        number_of_classes = model.number_of_classes
        _, axs = plt.subplots(1, number_of_classes, figsize=(6 * number_of_classes, 4))

    _plot_model_to_axes(model, title, axs)
    plt.tight_layout()
    plt.show()


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


def plot_motifs_and_length_distribution_per_group_mhc1(
    df_model_dirs: pd.DataFrame,
    output_directory: Openable,
    background: BackgroundType | None = None,
) -> None:
    """Plot the motifs of the MHC1 deconvolution models and the length distributions.

    For each gene (A, B, C etc.; if existent) and each available number of classes, one plot is
    created that contains the motifs for each class as well as the length distribution of the
    peptides that where assigned to the respective motif during the deconvolution (maximum
    responsibility).

    Args:
        df_model_dirs: DataFrame containing the following columns:
            - allele: Allele.
            - number_of_classes: Number of classes.
            - model_path: Path to the fitted deconvolution model.
        output_directory: Local or remote directory where to save the plots.
        background: Background frequencies to use. If None, the uniprot background is used.
    """
    output_directory = AnyPath(output_directory)
    df_model_dirs = df_model_dirs.copy()

    if background is None:
        background = Background("uniprot")
        log.info("Using uniprot background frequencies for plotting")
    else:
        log.info("Using custom background frequencies for plotting")

    try:
        # if the allele is a valid MHC1 allele, extract the gene
        df_model_dirs["gene"] = df_model_dirs["group"].apply(lambda x: parse_mhc1_allele_pair(x)[0])
        log.info("Extracted gene from group column")
    except ValueError:
        # use the groups as they are, set gene to "MHC1" for all such that the plots are not split
        df_model_dirs["gene"] = "MHC1"
        log.info("Could not extract gene from group column, not splitting plots")

    for (gene, number_of_classes), df_gene in df_model_dirs.groupby(["gene", "number_of_classes"]):
        log.info(
            f"Plotting deconvolution results: {gene}, "
            f"{number_of_classes} class{'es' if number_of_classes != 1 else ''}"
        )

        num_of_groups = len(df_gene)

        n_columns = 3 * number_of_classes + 2
        _, axs = plt.subplots(num_of_groups, n_columns, figsize=(6 * n_columns, 4 * num_of_groups))

        if num_of_groups == 1:
            axs = axs[np.newaxis, :]

        for i, (_, row) in enumerate(df_gene.iterrows()):
            group = str(row.group)
            log.info(f"Plotting {gene}, group {i+1}/{num_of_groups} ({group})")

            model_path = AnyPath(row.model_path)
            model = DeconvolutionModelMHC1.load(model_path)

            if model.number_of_classes != number_of_classes:
                raise ValueError(
                    f"number of classes in model {row.model_path} does not match, should be "
                    f"{number_of_classes}, got {model.number_of_classes}"
                )

            _plot_model_to_axes(model, group, axs[i, :-2:3], background)

            _plot_mhc1_class_weights_to_axes(
                model,
                group,
                list(axs[i, 1:-2:3]) + [axs[i, -2]],
                include_flat=True,
            )

            responsibilities = pd.read_csv(model_path / "responsibilities.csv")
            _plot_length_distribution_from_responsibilities(
                model,
                responsibilities,
                group,
                list(axs[i, 2:-2:3]) + [axs[i, -1]],
                include_flat=True,
            )

        _unify_ylim(axs[:, :-2:3])
        plt.tight_layout()

        file_path = output_directory / f"{gene}_classes_{number_of_classes}.pdf"
        save_plot(file_path)


def plot_motifs_and_length_distribution_per_group_mhc2(
    df_model_dirs: pd.DataFrame,
    output_directory: Openable,
) -> None:
    """Plot the motifs of the MHC2 deconvolution models and the length distributions.

    If the group column contains the alpha and beta chain alleles separated by a hyphen, then the
    plots are split by in the following way: For each gene (DR, DP, and DQ; if existent) and each
    available number of classes, one plot is created that contains the motifs for each class as
    well as the length distribution of the peptides that where assigned to the respective motif
    during the deconvolution (maximum responsibility). Otherwise, the plots are only split by the
    number of classes.

    Args:
        df_model_dirs: DataFrame containing the following columns:
            - group: Group; or alpha and beta chain allele separated by a hyphen.
            - number_of_classes: Number of classes.
            - model_path: Path to the fitted deconvolution model.
        output_directory: Local or remote directory where to save the plots.
    """
    output_directory = AnyPath(output_directory)
    df_model_dirs = df_model_dirs.copy()

    try:
        # try to parse alpha and beta chain alleles from the group column, and extract the gene
        df_model_dirs["gene"] = df_model_dirs["group"].apply(
            lambda x: parse_mhc2_allele_pair(x)[0][:2]
        )
        log.info("Extracted gene from group column")
    except ValueError:
        # use the groups as they are, set gene to "MHC2" for all such that the plots are not split
        df_model_dirs["gene"] = "MHC2"
        log.info("Could not extract gene from group column, not splitting plots")

    for (gene, number_of_classes), df_gene in df_model_dirs.groupby(["gene", "number_of_classes"]):
        log.info(
            f"Plotting deconvolution results: {gene}, "
            f"{number_of_classes} class{'es' if number_of_classes != 1 else ''}"
        )

        num_of_groups = len(df_gene)

        n_columns = 3 * number_of_classes + 2
        _, axs = plt.subplots(num_of_groups, n_columns, figsize=(6 * n_columns, 4 * num_of_groups))

        if num_of_groups == 1:
            axs = axs[np.newaxis, :]

        for i, (_, row) in enumerate(df_gene.iterrows()):
            group = str(row.group)
            log.info(f"Plotting {gene}, group {i+1}/{num_of_groups} ({group})")
            model_path = AnyPath(row.model_path)
            model = DeconvolutionModelMHC2.load(model_path)

            if model.number_of_classes != number_of_classes:
                raise ValueError(
                    f"number of classes in model {row.model_path} does not match, should be "
                    f"{number_of_classes}, got {model.number_of_classes}"
                )

            _plot_model_to_axes(model, group, axs[i, :-2:3])

            _plot_mhc2_offset_weights_to_axes(
                model,
                group,
                list(axs[i, 1:-2:3]) + [axs[i, -2]],
                include_flat=True,
            )

            responsibilities = pd.read_csv(model_path / "responsibilities.csv")
            _plot_length_distribution_from_responsibilities(
                model,
                responsibilities,
                group,
                list(axs[i, 2:-2:3]) + [axs[i, -1]],
                include_flat=True,
            )

        _unify_ylim(axs[:, :-2:3])
        plt.tight_layout()

        file_path = output_directory / f"{gene}_classes_{number_of_classes}.pdf"
        save_plot(file_path)


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


def _get_weights_per_motif(
    model: DeconvolutionModelMHC1 | DeconvolutionModelMHC2,
) -> np.ndarray | None:
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
    model: DeconvolutionModelMHC1 | DeconvolutionModelMHC2,
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
    model: DeconvolutionModelMHC1 | DeconvolutionModelMHC2,
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

        _title = f"length distribution ({_title}, weight {class_weights[i]:.3f})"
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
