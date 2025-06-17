"""Module to define functions for plotting contaminants."""
from __future__ import annotations

from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emmo.io.file import Openable
from emmo.utils import logger
from emmo.utils.viz.common import save_plot

log = logger.get(__name__)


def plot_contaminant_counts(
    df: pd.DataFrame,
    contaminants: list[dict[str, Any]],
    sort_by_total_contaminants: bool = True,
    num_experiments_per_plot: int = 50,
    save_as: Openable | None = None,
) -> None:
    """Plot the number of peptides and contaminants per experiment.

    This function creates a bar plot showing the number of peptides and contaminants for each
    experiment. The bars are stacked to show the contribution of each contaminant type.
    The plot is divided into subplots, with a specified number of experiments per subplot.

    Args:
        df: DataFrame containing peptides with contaminant annotations.
        contaminants: List of contaminant specifications.
        sort_by_total_contaminants: If True, sort by total contaminants. Otherwise, sort by total
            number of peptides.
        num_experiments_per_plot: Number of experiments to plot per subplot.
        save_as: If provided, save the plot at this location.
    """
    df_peptides_aggregated = _aggregate_contaminants_per_experiment(df, contaminants)

    total_proportion_of_contaminants = (
        df_peptides_aggregated["contaminant_any"].sum() / df_peptides_aggregated["peptide"].sum()
    )

    if len(contaminants) <= 10:
        colors = plt.cm.tab10.colors
    elif len(contaminants) <= 20:
        colors = plt.cm.tab20.colors
    else:
        colors = plt.cm.jet(np.linspace(0, 1, len(contaminants)))
        colors = [colors[i] for i in range(len(colors))]

    # remove experiments without any contaminants
    num_total_experiments = len(df_peptides_aggregated)
    df_peptides_aggregated = df_peptides_aggregated[df_peptides_aggregated["contaminant_any"] > 0]
    num_experiments_with_contaminants = len(df_peptides_aggregated)

    df_counts_sorted = df_peptides_aggregated.sort_values(
        "contaminant_any" if sort_by_total_contaminants else "peptide", ascending=False
    )

    num_plots = int(np.ceil(len(df_counts_sorted) / num_experiments_per_plot))

    _, axs = plt.subplots(
        num_plots,
        1,
        figsize=(2 + 0.25 * num_experiments_per_plot, 6 * num_plots),
        sharey=True,
        squeeze=False,
    )
    axs = axs.flatten()

    for i, ax in enumerate(axs):
        from_idx = i * num_experiments_per_plot
        to_idx = (i + 1) * num_experiments_per_plot

        # plot total number of peptides
        barplot = df_counts_sorted["peptide"][from_idx:to_idx].plot(
            kind="bar", ax=ax, color="lightgray", legend=True, label="total peptides"
        )

        for total_contaminants, bar in zip(
            df_counts_sorted["contaminant_any"][from_idx:to_idx], barplot.patches
        ):
            _annotate_bar_with_percent_contaminants(total_contaminants, bar, ax)

        # plot number of peptides with multiple contaminant annotations
        cumulated_contaminants = df_counts_sorted["contaminant_multiple"][from_idx:to_idx]
        cumulated_contaminants.plot(
            kind="bar",
            ax=ax,
            color="black",
            legend=True,
            label="multiple contaminant types",
        )

        # plot number of peptides with only one contaminant type
        for contaminant, color in zip(contaminants, colors):
            column = f"ONLY_contaminant_{contaminant['identifier']}"
            bar_heights = df_counts_sorted[column][from_idx:to_idx]
            bar_heights.plot(
                kind="bar",
                bottom=cumulated_contaminants,
                ax=ax,
                color=color,
                legend=True,
                label=contaminant["identifier"],
            )
            cumulated_contaminants = cumulated_contaminants + bar_heights

        ax.set_xlabel("")
        ax.set_ylabel("Num. of peptides")

        # ensure that percentages are shown inside of the plot
        ax.set_ylim(0, ax.get_ylim()[1] * 1.075)

        if i > 0:
            ax.get_legend().remove()

    legend_title = (
        "Contaminants per experiment"
        f" ({num_experiments_with_contaminants:,} out of"
        f" {num_total_experiments} experiments,"
        f" {total_proportion_of_contaminants:.2%} contaminants in total)"
    )
    axs[0].legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 1),
        title=legend_title,
        title_fontsize="13",
        fontsize="10",
        ncol=min(3, len(contaminants) + 1),
    )

    plt.tight_layout()

    if save_as is not None:
        save_plot(save_as)
    else:
        plt.show()


def _aggregate_contaminants_per_experiment(
    df: pd.DataFrame,
    contaminants: list[dict[str, Any]],
) -> pd.DataFrame:
    """Aggregate the number of contaminants per experiment.

    Args:
        df: DataFrame containing peptides with contaminant annotations.
        contaminants: List of contaminant specifications.

    Returns:
        DataFrame with aggregated contaminant counts per experiment.
    """
    df = df.copy()

    columns_to_aggregate = {"peptide": "count", "contaminant_any": "sum"}
    contaminant_columns = [
        f"contaminant_{contaminant['identifier']}" for contaminant in contaminants
    ]
    for column in contaminant_columns:
        columns_to_aggregate[column] = "sum"

    # handle peptides that have multiple contaminant annotation such that they are not counted twice
    df["contaminant_multiple"] = (df[contaminant_columns].sum(axis=1) > 1).astype(int)
    columns_to_aggregate["contaminant_multiple"] = "sum"
    for column in contaminant_columns:
        new_column = f"ONLY_{column}"
        df[new_column] = (df[column] & ~df["contaminant_multiple"]).astype(int)
        columns_to_aggregate[new_column] = "sum"

    df_peptides_aggregated = df.groupby("group").agg(columns_to_aggregate)

    return df_peptides_aggregated


def _annotate_bar_with_percent_contaminants(
    total_contaminants: int, bar: matplotlib.patches.Rectangle, ax: matplotlib.axes._axes.Axes
) -> None:
    """Annotate the bar with the percentage of contaminants.

    Args:
        total_contaminants: The total number of contaminants.
        bar: The bar to annotate (corresponding to the total number of peptides).
        ax: The axes of the bar plot.
    """
    x = bar.get_x() + bar.get_width() / 2
    y = bar.get_height()
    distance = (ax.get_ylim()[1] - ax.get_ylim()[0]) / 50

    percent_contaminated = 100 * total_contaminants / y
    annotation = f"{percent_contaminated:.2f} %"

    ax.text(
        x,
        y + distance,
        annotation,
        rotation=90,
        horizontalalignment="center",
        verticalalignment="bottom",
    )
