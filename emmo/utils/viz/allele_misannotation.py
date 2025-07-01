"""Module to define functions and classes for plotting allele misannotations."""
from __future__ import annotations

import math
from typing import Any
from typing import Iterator

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.axes import Axes
from matplotlib.figure import Figure

from emmo.constants import EPSILON
from emmo.io.file import Openable
from emmo.models.deconvolution import DeconvolutionModelMHC2
from emmo.pipeline.background import Background
from emmo.utils import logger
from emmo.utils.alleles import parse_mhc1_allele_pair
from emmo.utils.alleles import split_and_shorten_alleles
from emmo.utils.sequence_distance import nearest_neighbors
from emmo.utils.viz.common import COLOR_DEFAULT
from emmo.utils.viz.common import COLOR_DEFAULT_LIGHT
from emmo.utils.viz.common import is_axes_empty
from emmo.utils.viz.common import rectangle_with_text
from emmo.utils.viz.common import save_plot
from emmo.utils.viz.common import unify_ylim
from emmo.utils.viz.motifs_and_deconvolution import _plot_deconvolution_run_to_axes
from emmo.utils.viz.motifs_and_deconvolution import plot_single_ppm

log = logger.get(__name__)


COLOR_IN_REFERENCE = "grey"
COLOR_NOT_IN_REFERENCE = "indianred"


class MotifDistancePlotter:
    """Class to plot the motif distances for a set of groups (single page)."""

    def __init__(
        self,
        df_distances: pd.DataFrame,
        df_hit_counts: pd.DataFrame | None = None,
        used_pct_rank_threshold: float | None = None,
        allele_column: str | None = None,
        alleles_in_reference: set[str] | dict[str, Any] | None = None,
        ylim_distances: int | None = None,
        ylim_hit_counts: int | None = None,
    ) -> None:
        """Initialize the MotifDistancePlotter.

        Args:
            df_distances: DataFrame with the distances to the best matching reference motifs. The
                DataFrame should contain the following columns:
                - `group`: Group identifier (e.g., experiment ID).
                - `best_matching_allele_genotype`: Best matching allele in the provided set of
                   alleles (i.e., in the genotype).
                - `best_matching_distance_genotype`: Distance to the best matching allele in the
                   provided set of alleles.
                - `best_matching_allele_is_in_reference_genotype`: Whether the best matching allele
                   in the genotype is in the set of reference alleles.
                - `best_matching_allele_overall`: Best matching allele across all reference alleles.
                - `best_matching_distance_overall`: Distance to the overall best matching allele.
                - `best_matching_allele_is_in_reference_overall`: Whether the overall best matching
                   allele is in the set of reference alleles.
            df_hit_counts: DataFrame with the hit counts for each group. If None, no hit counts are
                plotted. The DataFrame should contain the following columns:
                - `group`: Group identifier (e.g., experiment ID).
                - `num_of_hits_total`: Total number of hits for the group.
                - `num_of_hits_below_threshold`: Number of hits that satisfy some threshold
                  (e.g., below a certain percentile rank).
            used_pct_rank_threshold: Percentile rank threshold that was used to filter the hit
                counts. Only used to have this information in the plot legend.
            allele_column: Name of the column in `df_distances` that contains the alleles for each
                group. If None, no alleles are plotted.
            alleles_in_reference: Set or dictionary of alleles that are in the reference. If None,
                no distinction is made between alleles in terms of the color in the allele part of
                the plot.
            ylim_distances: Optional y-axis limit for the distances plot. If None, the limits are
                automatically determined based on the data.
            ylim_hit_counts: Optional y-axis limit for the hit counts plot. If None, the limits are
                automatically determined based on the data.
        """
        # check that the DataFrame has the expected columns
        expected_columns = {
            "group",
            "best_matching_allele_genotype",
            "best_matching_distance_genotype",
            "best_matching_allele_is_in_reference_genotype",
            "best_matching_allele_overall",
            "best_matching_distance_overall",
            "best_matching_allele_is_in_reference_overall",
        }
        if not expected_columns.issubset(df_distances.columns):
            raise ValueError(
                "DataFrame is missing expected columns: "
                f"{sorted(expected_columns - set(df_distances.columns))}"
            )
        self.df_distances = df_distances

        if df_hit_counts is not None:
            self.df_hit_counts = self._prepare_hit_counts(df_hit_counts)
        else:
            self.df_hit_counts = None

        self.used_pct_rank_threshold = used_pct_rank_threshold
        self.allele_column = allele_column
        self.alleles_in_reference = alleles_in_reference
        self.ylim_distances = ylim_distances
        self.ylim_hit_counts = ylim_hit_counts

        # initialize the properties that will be set later
        self._x_ticks: list[float] | None = None
        self._x_labels: list[str] | None = None
        self._box_centers: list[float] | None = None
        self._xlims: tuple[float, float] | None = None

    @property
    def x_ticks(self) -> list[float]:
        """Get the x ticks of the boxplot."""
        if self._x_ticks is None:
            raise ValueError("x ticks have not been set yet")

        return self._x_ticks

    @property
    def x_labels(self) -> list[str]:
        """Get the x labels of the boxplot, which are the group identifiers."""
        if self._x_labels is None:
            raise ValueError("x labels have not been set yet")

        return self._x_labels

    @property
    def box_centers(self) -> list[float]:
        """Get the centers of the boxes in the boxplot."""
        if self._box_centers is None:
            raise ValueError("box centers have not been set yet")

        return self._box_centers

    @property
    def box_width(self) -> float:
        """Get the width of the boxes in the boxplot."""
        return self.box_centers[1] - self.box_centers[0]

    @property
    def xlims(self) -> tuple[float, float]:
        """Get the x limits of the boxplot."""
        if self._xlims is None:
            raise ValueError("x limits have not been set yet")

        return self._xlims

    def plot(self) -> Figure:
        """Plot the motif distances.

        Plots the Jensen-Shannon divergence to the best reference motifs (in the genotype vs
        overall as paired boxplot), and optionally the hit counts and the alleles present in the
        genotype.

        Returns:
            The matplotlib Figure object containing the plot.
        """
        weights = []

        if self.allele_column is not None:
            weights.append(0.5)

        if self.df_hit_counts is not None:
            weights.append(0.8)

        # weight of the motif distances plot is always 1
        weights.append(1)

        num_plots = len(weights)

        fig, ax = plt.subplots(
            num_plots,
            1,
            figsize=(20, 5 * sum(weights)),
            gridspec_kw={"height_ratios": weights},
            squeeze=False,
        )

        self._plot_motif_distances(ax[-1, 0])
        if self.ylim_distances is not None:
            ax[-1, 0].set_ylim(0, self.ylim_distances)

        if self.allele_column is not None:
            self._plot_alleles_in_genotype(ax[0, 0])

        if self.df_hit_counts is not None:
            self._plot_hit_counts(ax[-2, 0])
            if self.ylim_hit_counts is not None:
                ax[-2, 0].set_ylim(0, self.ylim_hit_counts)

        for i in range(num_plots):
            ax[i, 0].set_xlim(*self.xlims)

            if i < num_plots - 1:
                ax[i, 0].set_xticklabels([])

        if num_plots > 1:
            # repeat the x ticks and labels for at the top of the first plot
            ax[0, 0].xaxis.set_ticks_position("both")
            ax[0, 0].xaxis.set_label_position("top")
            ax[0, 0].set_xticks(self.x_ticks)
            ax[0, 0].set_xticklabels([])
            ax_secondary = ax[0, 0].twiny()
            ax_secondary.set_xlim(ax[0, 0].get_xlim())
            ax_secondary.set_xticks(self.x_ticks)
            ax_secondary.set_xticklabels(self.x_labels, rotation=90)

        plt.tight_layout()

        return fig

    def _prepare_hit_counts(self, df_hit_counts: pd.DataFrame) -> pd.DataFrame:
        """Prepare the hit counts DataFrame for plotting.

        The groups should be in the same order as in the distances DataFrame. Moreover, it checks
        that all groups in the distances DataFrame have a corresponding entry in the hit counts
        DataFrame.

        Args:
            df_hit_counts: DataFrame with the hit counts for each group.

        Returns:
            DataFrame with the hit counts in the same order as the distances DataFrame.

        Raises:
            ValueError: If there are groups in the distances DataFrame that do not have a
                corresponding entry in the hit counts DataFrame (or if there are NaN values in the
                'num_of_hits_total' column).
        """
        expected_columns = {"group", "num_of_hits_total", "num_of_hits_below_threshold"}
        if not expected_columns.issubset(df_hit_counts.columns):
            raise ValueError(
                "DataFrame is missing expected columns: "
                f"{sorted(expected_columns - set(df_hit_counts.columns))}"
            )

        df_merged = pd.merge(
            self.df_distances,
            df_hit_counts,
            on="group",
            how="left",
        )

        num_nans = df_merged["num_of_hits_total"].isna().sum()
        if num_nans > 0:
            raise ValueError(f"found {num_nans} NaN values in the 'num_of_hits_total' column")

        df_dedup = df_merged[df_hit_counts.columns].drop_duplicates("group", ignore_index=True)

        return df_dedup

    def _melt_best_matching_alleles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Melt the best matching allele from the genotype and overall best matching allele.

        Args:
            df: DataFrame with the best matching alleles from the genotype and overall in separate
                columns.

        Returns:
            Melted DataFrame with the best matching alleles.
        """
        # combine the two columns that belong together into a single column
        best_match_genotype = df.apply(
            lambda row: (
                row["best_matching_allele_genotype"],
                row["best_matching_allele_is_in_reference_genotype"],
                row["best_matching_distance_genotype"],
            ),
            axis=1,
        )
        best_match_overall = df.apply(
            lambda row: (
                row["best_matching_allele_overall"],
                row["best_matching_allele_is_in_reference_overall"],
                row["best_matching_distance_overall"],
            ),
            axis=1,
        )

        # melt the columns "best_match_genotype" and "best_match_overall" into a single column
        # "best_match", with the corresponding "matching_type"
        df_melted = df.assign(
            best_match_genotype=best_match_genotype,
            best_match_overall=best_match_overall,
        ).melt(
            id_vars=["group"],
            value_vars=["best_match_genotype", "best_match_overall"],
            var_name="matching_type",
            value_name="best_match",
        )

        # separate the best match again into two columns
        df_melted[
            [
                "best_matching_allele",
                "best_matching_allele_is_in_reference",
                "best_matching_distance",
            ]
        ] = pd.DataFrame(df_melted["best_match"].tolist(), index=df_melted.index)
        df_melted = df_melted.drop(columns=["best_match"])

        return df_melted

    def _plot_motif_distances(self, ax: plt.Axes) -> None:
        """Plot the Jensen-Shannon divergence to the best reference motifs.

        This produces one boxplot per group with the distances to the best matching reference motifs
        among the available alleles in the dataset. The boxplots are grouped by the matching type,
        i.e., whether the best matching allele is the best in the genotype or the best overall.

        Args:
            df: DataFrame with the distances to the best matching reference motifs.
            ax: Matplotlib Axes object to plot on. If None, a new figure and axes are created.
        """
        if ax is None:
            _, axs = plt.subplots(1, 1, figsize=(20, 5), squeeze=False)
            ax = axs[0, 0]

        df_melted = self._melt_best_matching_alleles(self.df_distances)

        sns.boxplot(
            data=df_melted.assign(
                matching_type=df_melted["matching_type"].map(
                    {
                        "best_match_genotype": "Best in genotype",
                        "best_match_overall": "Best overall",
                    }
                )
            ),
            x="group",
            y="best_matching_distance",
            hue="matching_type",
            showfliers=False,
            palette=sns.color_palette("Paired")[:2],
            boxprops={
                "linewidth": 1.0,
                "edgecolor": "black",
            },
            medianprops={
                "linewidth": 1.5,
                "color": "black",
            },
            whiskerprops={
                "linewidth": 1.0,
                "color": "black",
            },
            ax=ax,
        )

        self._set_x_axis_labels(ax)
        self._add_individual_alleles_to_distance_boxplot(df_melted, ax)

        ax.tick_params(axis="x", labelrotation=90)
        ax.set_xlabel("")
        ax.set_ylabel("JS divergence to best allele")
        ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), ncol=1).set_title(None)

    def _set_x_axis_labels(self, ax: plt.Axes) -> None:
        """Set the x ticks and labels for the boxplot.

        Args:
            ax: Matplotlib Axes object to set the x ticks and labels on.

        Raises:
            ValueError: If the number of x ticks and x labels does not match, or if the number of
                box centers does not match the number of x labels.
        """
        self._x_ticks = ax.get_xticks()
        self._x_labels = [label.get_text() for label in ax.get_xticklabels()]

        if len(self._x_ticks) != len(self._x_labels):
            raise ValueError(
                "The number of x ticks and x labels does not match. "
                f"Got {len(self._x_ticks)} ticks and {len(self._x_labels)} labels."
            )

        # get the centers of the boxes to use them for the scatter plot
        box_centers = []
        for box in ax.patches:
            if box.get_path().vertices.shape[0] != 6:
                continue

            center = box.get_path().vertices[:2, 0].mean()
            box_centers.append(center)

        self._box_centers = sorted(box_centers)

        if len(self._box_centers) != 2 * len(self._x_labels):
            raise ValueError(
                "The number of box centers does not match the number of x labels. "
                f"Got {len(self._box_centers)} centers and {len(self._x_labels)} labels."
            )

        self._xlims = (
            self._box_centers[0] - self.box_width,
            self._box_centers[-1] + self.box_width,
        )

    def _add_individual_alleles_to_distance_boxplot(
        self,
        df_melted: pd.DataFrame,
        ax: plt.Axes,
    ) -> None:
        """Add individual alleles to the distance boxplot as scatter points.

        The scatter points are colored based on whether the allele is in the reference or not.

        Args:
            df_melted: Melted DataFrame with the best matching alleles.
            ax: Matplotlib Axes object to plot on.

        Raises:
            ValueError: If the scatter plot data does not have the same length as the original data.
        """
        data_for_scatterplot = []
        for i, group in enumerate(self.x_labels):
            data_for_scatterplot.append(
                {
                    "group": group,
                    "box_center": self.box_centers[2 * i],
                    "matching_type": "best_match_genotype",
                }
            )
            data_for_scatterplot.append(
                {
                    "group": group,
                    "box_center": self.box_centers[2 * i + 1],
                    "matching_type": "best_match_overall",
                }
            )
        df_scatter = pd.merge(
            df_melted[
                [
                    "group",
                    "matching_type",
                    "best_matching_distance",
                    "best_matching_allele_is_in_reference",
                ]
            ],
            pd.DataFrame(data_for_scatterplot),
            on=["group", "matching_type"],
            how="left",
        )
        if len(df_scatter) != len(df_melted):
            raise ValueError(
                "the scatter plot data should have the same length as the original data"
            )

        df_scatter["best_matching_allele_is_in_reference"] = df_scatter[
            "best_matching_allele_is_in_reference"
        ].map({True: "In reference", False: "Not in reference"})

        sns.scatterplot(
            data=df_scatter,
            x="box_center",
            y="best_matching_distance",
            hue="best_matching_allele_is_in_reference",
            palette={
                "In reference": COLOR_IN_REFERENCE,
                "Not in reference": COLOR_NOT_IN_REFERENCE,
            },
            markers="o",
            s=30,
            linewidth=0,  # disable the border around the markers
            zorder=10,
            ax=ax,
        )

    def _plot_hit_counts(self, ax: plt.Axes) -> None:
        """Plot the hit counts per group.

        This plots both the total number of hits and only the hits below the percentile rank
        threshold.

        Args:
            ax: Matplotlib Axes object to plot on.

        Raises:
            ValueError: If the x ticks of the hit counts plot do not match the x ticks of the
                distances plot.
        """
        df = self.df_hit_counts.set_index("group", drop=True)

        plot = df["num_of_hits_total"].plot(
            kind="bar",
            ax=ax,
            color=COLOR_DEFAULT_LIGHT,
            legend=True,
            label="total hit count",
        )

        # assert that the x ticks are in the expected order
        if list(plot.get_xticks()) != list(self.x_ticks):
            raise ValueError(
                "The x ticks of the hit counts plot do not match the x ticks of the distances plot,"
                f" got {list(plot.get_xticks())} but expected {self.x_ticks}."
            )

        df["num_of_hits_below_threshold"].plot(
            kind="bar",
            ax=ax,
            color=COLOR_DEFAULT,
            legend=True,
            label=(
                f"hits <={self.used_pct_rank_threshold}%\npercentile rank"
                if self.used_pct_rank_threshold is not None
                else "hits below threshold"
            ),
        )

        ax.set_xlabel("")
        ax.set_ylabel("Num. of unique peptides")
        ax.legend(loc="upper left", bbox_to_anchor=(1.005, 1.0), ncol=1)

    def _plot_alleles_in_genotype(self, ax: plt.Axes) -> None:
        """Plot the alleles per group.

        Args:
            ax: Matplotlib Axes object to plot on.

        Raises:
            ValueError: If there more unique genotypes than groups.
        """
        df_alleles = self.df_distances[["group", self.allele_column]].drop_duplicates(
            ignore_index=True
        )

        if len(df_alleles) != len(self.x_ticks):
            raise ValueError("there should be one genotype per group")

        for i, alleles in enumerate(df_alleles[self.allele_column]):
            alleles = sorted(
                {
                    parse_mhc1_allele_pair(
                        allele.replace("HLA-", "").replace("*", "").replace(":", "")
                    )
                    for allele in alleles.replace(";", ",").replace(" ", "").split(",")
                }
            )
            locus2alleles: dict[str, list[str]] = {"A": [], "B": [], "C": []}
            for allele in alleles:
                locus = allele[0]

                if locus not in locus2alleles:
                    raise ValueError(
                        f"Unexpected locus '{locus}' in allele '{allele}'. "
                        "Expected one of 'A', 'B', or 'C'."
                    )

                locus2alleles[locus].append(allele)

            self._plot_alleles_in_genotype_single_group(locus2alleles, self.x_ticks[i], ax)

        ax.set_xticks(self.x_ticks)
        ax.set_yticks([])

    def _plot_alleles_in_genotype_single_group(
        self,
        locus2alleles: dict[str, list[str]],
        tick_position: float,
        ax: plt.Axes,
    ) -> None:
        """Plot the alleles in the genotype for a single group.

        Args:
            locus2alleles: Dictionary mapping loci to lists of alleles for that locus.
            tick_position: The x position of the tick for the group.
            ax: Matplotlib Axes object to plot on.
        """
        for locus_idx, alleles in enumerate(locus2alleles.values()):
            for j, allele in enumerate(alleles):
                x = tick_position - self.box_width + j * (2 * self.box_width / len(alleles))
                y = locus_idx / len(locus2alleles)
                width = 2 * self.box_width / len(alleles)
                height = 1 / len(locus2alleles)

                facecolor = COLOR_IN_REFERENCE

                if (
                    self.alleles_in_reference is not None
                    and allele not in self.alleles_in_reference
                ):
                    facecolor = COLOR_NOT_IN_REFERENCE

                rectangle_with_text(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    ax=ax,
                    facecolor=facecolor,
                    text=allele,
                    fontsize=10,
                    text_color="white",
                    text_rotation=90,
                )


class MotifComparisonPlotter:
    """Class to plot the deconvoluted motifs and the closest matching reference motifs."""

    def __init__(
        self,
        df_distances: pd.DataFrame,
        allele2reference_motif: dict[str, np.ndarray],
        mhc_class: int,
        allele_column: str | None = None,
    ) -> None:
        """Initialize the MotifComparisonPlotter.

        Args:
            df_distances: DataFrame with the distances to the best matching reference motifs. The
                DataFrame should contain the following columns:
                - `group`: Group identifier (e.g., experiment ID).
                - `number_of_classes`: Number of classes in the deconvolution run.
                - `model_path`: Path to the deconvolution model used for the run.
                - `class`: Class/cluster index in the deconvolution run (1-based).
                - `best_matching_allele_genotype`: Best matching allele in the provided set of
                   alleles (i.e., in the genotype).
                - `best_matching_distance_genotype`: Distance to the best matching allele in the
                   provided set of alleles (i.e., in the genotype).
                - `best_matching_allele_overall`: Best matching allele across all reference alleles.
                - `best_matching_distance_overall`: Distance to the overall best matching allele.
            allele2reference_motif: Dictionary mapping alleles to their corresponding reference
                position probability matrices (PPMs).
            mhc_class: MHC class for which to plot the motifs (1 or 2).
            allele_column: Optional name of the column in `df_distances` that contains the alleles
                for each group. If None, the reference motifs of those alleles are not plotted.

        Raises:
            ValueError: If the DataFrame is missing expected columns or if the MHC class is not 1
                or 2.
        """
        # check that the DataFrame has the expected columns
        expected_columns = {
            "group",
            "number_of_classes",
            "model_path",
            "class",
            "best_matching_allele_genotype",
            "best_matching_distance_genotype",
            "best_matching_allele_overall",
            "best_matching_distance_overall",
        }
        if allele_column is not None:
            expected_columns.add(allele_column)
        if not expected_columns.issubset(df_distances.columns):
            raise ValueError(
                "DataFrame is missing expected columns: "
                f"{sorted(expected_columns - set(df_distances.columns))}"
            )

        if mhc_class not in (1, 2):
            raise ValueError("MHC class must be either 1 or 2.")

        if mhc_class == 2:
            raise NotImplementedError(
                "MHC class 2 is not yet implemented in the MotifComparisonPlotter."
            )

        self.df_distances = df_distances
        self.allele2reference_motif = allele2reference_motif
        self.mhc_class = mhc_class
        self.allele_column = allele_column

        self._allele2nearest_reference_allele = self._map_alleles_to_nearest_reference()

        # used to unify the y-axis limits of the subplots
        self._axs_to_normalize: list[Axes] = []

    def plot(
        self,
        group: str | list[str],
        save_as: Openable | None = None,
        disable_display: bool = False,
    ) -> None:
        """Plot the deconvoluted motifs and the closest matching reference motifs.

        Args:
            group: The identifier of the group to plot (e.g., an experiment ID). A list of groups
                can also be provided, in which case the groups are plotted as pages of a PDF file.
            save_as: Optional file path to save the plot. If None, the plot is not saved.
            disable_display: If True (and `save_as` is provided), the plot is not displayed
                interactively.

        Returns:
            The matplotlib Figure object containing the plot.

        Raises:
            ValueError: If the group identifier is not found in the DataFrame.
        """
        if save_as is None and disable_display:
            raise ValueError(
                "If 'disable_display' is True, 'save_as' must be provided to save the plot."
            )

        groups = group if isinstance(group, list) else [group]

        figures: list[Figure] = []

        for i, this_group in enumerate(groups):
            if isinstance(group, list):
                log.info(f"Plotting page {i + 1} of {len(groups)} ('{this_group}') ...")

            figures.append(self._plot(this_group))

        if save_as is not None:
            save_plot(save_as, figures=figures, close_figures=disable_display)

    def _filter_by_group(self, group: str) -> pd.DataFrame:
        """Filter the DataFrame by the group identifier.

        Args:
            group: The identifier of the group to filter by (e.g., an experiment ID).

        Returns:
            Filtered DataFrame containing only the rows for the specified group.

        Raises:
            ValueError: If the group identifier is not found in the DataFrame or if there are
                inconsistencies in the number of classes, classes indexes, or model paths.
        """
        df_filtered = (
            self.df_distances[self.df_distances["group"] == group]
            .sort_values("number_of_classes", ascending=True)
            .reset_index(drop=True)
        )

        if df_filtered.empty:
            raise ValueError(f"Identifier '{group}' not found in the DataFrame")

        # check that the number of classes are consistent
        number_of_classes = len(df_filtered)
        if [number_of_classes] != df_filtered["number_of_classes"].unique().tolist():
            raise ValueError(
                f"Expected {number_of_classes} classes, but found "
                f"{df_filtered['number_of_classes'].unique().tolist()}."
            )

        # check that the classes are in the expected range
        if df_filtered["class"].tolist() != list(range(1, number_of_classes + 1)):
            raise ValueError(
                f"Expected classes to be {list(range(1, number_of_classes + 1))}, "
                f"but found {df_filtered['class'].tolist()}."
            )

        # check that the model path is consistent
        model_paths = df_filtered["model_path"].unique()
        if len(model_paths) != 1:
            raise ValueError(
                f"Expected exactly one model path, but found {len(model_paths)} different ones"
            )

        return df_filtered

    def _map_alleles_to_nearest_reference(self) -> dict[str, tuple[list[str], float]]:
        """Map allele that are missing in the reference to the nearest reference allele.

        Returns:
            Dictionary mapping alleles that are not in the reference to the nearest reference allele
            and the distance to it.
        """
        all_alleles = set(self.df_distances["best_matching_allele_genotype"].unique()) | set(
            self.df_distances["best_matching_allele_overall"].unique()
        )

        if self.allele_column is not None:
            all_alleles = all_alleles.union(
                self.df_distances[self.allele_column]
                .apply(split_and_shorten_alleles)
                .explode()
                .unique()
            )

        reference_alleles = sorted(self.allele2reference_motif.keys())
        unavailable_alleles = sorted(all_alleles.difference(reference_alleles))

        return nearest_neighbors(
            available_alleles=reference_alleles,
            unavailable_alleles=unavailable_alleles,
            mhc_class=self.mhc_class,
        )

    def _plot(self, group: str) -> Figure:
        """Plot the deconvoluted motifs and the closest matching reference motifs for a group.

        Args:
            group: The identifier of the group to plot (e.g., an experiment ID).

        Returns:
            The matplotlib Figure object containing the plot.
        """
        df_filtered = self._filter_by_group(group)
        number_of_classes = df_filtered["number_of_classes"].iloc[0]
        model_path = str(df_filtered["model_path"].iloc[0])

        if self.mhc_class == 1:
            background = Background("uniprot")
        else:
            background = DeconvolutionModelMHC2.load(model_path).background

        n_classes = number_of_classes + 1  # +1 for the flat motif
        n_alleles = None

        if self.allele_column is None:
            nrows = n_classes
            ncols = 5
            num_subfigs = 1
            width_ratios = None
        else:
            alleles = sorted(
                set(split_and_shorten_alleles(df_filtered[self.allele_column].iloc[0]))
            )
            n_alleles = len(alleles)
            nrows = max(n_classes, n_alleles)
            ncols = 6
            num_subfigs = 2
            width_ratios = [1, 5]

        fig = plt.figure(layout="constrained", figsize=(6 * ncols, 4 * nrows))
        subfigs = fig.subfigures(
            1,
            num_subfigs,
            width_ratios=width_ratios,
            wspace=0.07,
            squeeze=False,
        )

        axs_deconv = subfigs[0, -1].subplots(nrows, 5, squeeze=False, gridspec_kw={"wspace": 0.05})
        all_axs = axs_deconv.flatten().tolist()

        self._plot_deconvolution_run_and_matched_motifs(
            axs_deconv[:n_classes, :],
            df_filtered,
            background,
        )

        if self.allele_column is not None:
            subfigs[0, 0].set_facecolor("cornsilk")

            # plot the alleles in the genotype
            axs_alleles = subfigs[0, 0].subplots(nrows, 1, gridspec_kw={"wspace": 0.01})
            all_axs.extend(axs_alleles.flatten().tolist())
            self._plot_alleles(axs_alleles[:n_alleles], alleles, background)

        # unify the y-axis limits of the subplots that contain motifs
        unify_ylim(self._axs_to_normalize)
        self._axs_to_normalize.clear()

        for ax in all_axs:
            if is_axes_empty(ax):
                ax.set_visible(False)

        return fig

    def _plot_deconvolution_run_and_matched_motifs(
        self,
        axs: np.ndarray[Axes],
        df_filtered: pd.DataFrame,
        background: Background,
    ) -> None:
        """Plot the deconvolution run and the closest matching reference motifs.

        Args:
            axs: Matplotlib Axes object to plot on.
            df_filtered: Filtered DataFrame containing the deconvolution run information for the
                specified group.
            background: Background amino acid frequencies to use for the motifs.
        """
        group = str(df_filtered["group"].iloc[0])
        model_path = str(df_filtered["model_path"].iloc[0])
        number_of_classes = int(df_filtered["number_of_classes"].iloc[0])

        _plot_deconvolution_run_to_axes(
            mhc_class=self.mhc_class,
            title=group,
            model_path=model_path,
            background=background,
            axs_motifs=axs[:-1, 0],
            axs_weights=axs[:, 1],
            axs_length=axs[:, 2],
            expected_number_of_classes=number_of_classes,
        )
        self._axs_to_normalize.extend(axs[:-1, 0].tolist())

        for suffix, j in zip(["genotype", "overall"], [3, 4]):
            for i, row in df_filtered.iterrows():
                allele = row[f"best_matching_allele_{suffix}"]
                distance = row[f"best_matching_distance_{suffix}"]
                ax = axs[i, j]

                if allele in self.allele2reference_motif:
                    ref_allele = allele
                    title = f"Closest motif {suffix}: {allele}\nJS divergence: {distance:.3f}"
                else:
                    nearest_reference_alleles = self._allele2nearest_reference_allele[allele][0]

                    # use the first nearest reference allele for plotting
                    ref_allele = nearest_reference_alleles[0]

                    # but write all nearest reference alleles in the title
                    alleles_str = ", ".join(nearest_reference_alleles)
                    title = (
                        f"Closest motif {suffix}: {allele} (nearest ref.: {alleles_str})\n"
                        f"JS divergence: {distance:.3f}"
                    )

                ppm = self.allele2reference_motif[ref_allele]
                plot_single_ppm(ppm, background=background, ax=ax)
                ax.set_title(title)
                self._axs_to_normalize.append(ax)

    def _plot_alleles(
        self,
        axs: np.ndarray[Axes],
        alleles: list[str],
        background: Background,
    ) -> None:
        """Plot the alleles in the genotype.

        Args:
            axs: Matplotlib Axes object to plot on.
            alleles: List of alleles to plot.
            background: Background amino acid frequencies to use for the motifs.
        """
        for i, allele in enumerate(alleles):
            ax = axs[i]

            if allele in self.allele2reference_motif:
                ref_allele = allele
                title = f"Allele {allele}"
            else:
                nearest_reference_alleles = self._allele2nearest_reference_allele[allele][0]

                # use the first nearest reference allele for plotting
                ref_allele = nearest_reference_alleles[0]

                # but write all nearest reference alleles in the title
                alleles_str = ", ".join(nearest_reference_alleles)
                title = f"Allele: {allele} (nearest ref.: {alleles_str})"

            ppm = self.allele2reference_motif[ref_allele]
            plot_single_ppm(ppm, background=background, ax=ax)
            ax.set_title(title)
            self._axs_to_normalize.append(ax)


def plot_group_motif_distances(
    df_distances: pd.DataFrame,
    file_path: Openable,
    df_hit_counts: pd.DataFrame | None = None,
    max_groups_per_page: int = 30,
    used_pct_rank_threshold: float | None = None,
    allele_column: str | None = None,
    alleles_in_reference: set[str] | dict[str, Any] | None = None,
    sort_by_distance: str | None = None,
    sort_metric: str = "median",
    sort_ascending: bool = False,
) -> None:
    """Plot the motif distances for all groups.

    Args:
        df_distances: DataFrame with the distances to the best matching reference motifs. The
            DataFrame should contain the following columns:
            - `group`: Group identifier (e.g., experiment ID).
            - `best_matching_allele_genotype`: Best matching allele in the provided set of alleles
                (i.e., in the genotype).
            - `best_matching_distance_genotype`: Distance to the best matching allele in the
                provided set of alleles.
            - `best_matching_allele_is_in_reference_genotype`: Whether the best matching allele in
                the genotype is in the set of reference alleles.
            - `best_matching_allele_overall`: Best matching allele across all reference alleles.
            - `best_matching_distance_overall`: Distance to the overall best matching allele.
            - `best_matching_allele_is_in_reference_overall`: Whether the overall best matching
                allele is in the set of reference alleles.
        file_path: Path to save the plot to.
        df_hit_counts: DataFrame with the hit counts for each group. If None, no hit counts are
            plotted. The DataFrame should contain the following columns:
            - `group`: Group identifier (e.g., experiment ID).
            - `num_of_hits_total`: Total number of hits for the group.
            - `num_of_hits_below_threshold`: Number of hits that satisfy some threshold (e.g., below
                a certain percentile rank).
        max_groups_per_page: Maximum number of groups to plot per page. If there are more groups,
            multiple pages will be created.
        used_pct_rank_threshold: Percentile rank threshold that was used to filter the hit counts.
            Only used to have this information in the plot legend.
        allele_column: Name of the column in `df_distances` that contains the alleles for each
            group. If None, no alleles are plotted.
        alleles_in_reference: Set or dictionary of alleles that are in the reference. If None, no
            distinction is made between alleles in terms of the color in the allele part of the
            plot.
        sort_by_distance: If not None, sort the groups by the motif distance. The value can be
            either 'ratio', 'genotype', or 'overall'.
        sort_metric: The metric to use for sorting, either 'median' or 'mean'.
        sort_ascending: Whether to sort in ascending order.
    """
    if df_distances.empty:
        log.warning("The DataFrame with distances is empty. No groups to plot.")
        return

    # sort the groups by the motif distance if requested
    if sort_by_distance is not None:
        log.info(
            f"Sorting groups by distance (mode: {sort_by_distance}) using {sort_metric} in "
            f"{'ascending' if sort_ascending else 'descending'} order."
        )
        df_distances = _sort_groups_by_motif_distance(
            df=df_distances,
            mode=sort_by_distance,
            metric=sort_metric,
            ascending=sort_ascending,
        )

    # use the generator function to avoid opening to many figures at once
    figures = _plot_group_motif_distances(
        df_distances,
        df_hit_counts,
        max_groups_per_page,
        used_pct_rank_threshold,
        allele_column,
        alleles_in_reference,
    )

    save_plot(file_path, figures)


def _plot_group_motif_distances(
    df_distances: pd.DataFrame,
    df_hit_counts: pd.DataFrame | None,
    max_groups_per_page: int,
    used_pct_rank_threshold: float | None,
    allele_column: str | None,
    alleles_in_reference: set[str] | dict[str, Any] | None,
) -> Iterator[Figure]:
    """Plot the motif distances for all groups.

    This function generates a plot for each page of groups, where each page contains a maximum of
    `max_groups_per_page` groups. The function yields the plot for each page.

    Args:
        df_distances: DataFrame with the distances to the best matching reference motifs.
        df_hit_counts: DataFrame with the hit counts for each group. If None, no hit counts are
            plotted.
        max_groups_per_page: Maximum number of groups to plot per page. If there are more
            groups, multiple pages will be created.
        used_pct_rank_threshold: Percentile rank threshold that was used to filter the hit counts.
            Only used to have this information in the plot legend.
        allele_column: Name of the column in `df_distances` that contains the alleles for each
            group. If None, no alleles are plotted.
        alleles_in_reference: Set or dictionary of alleles that are in the reference. If None, no
            distinction is made between alleles in terms of the color in the allele part of the
            plot.

    Yields:
        Figure: The plot for each page of groups.

    Raises:
        ValueError: If there are no groups in the DataFrame.
    """
    if df_distances.empty:
        raise ValueError("The DataFrame with distances is empty. No groups to plot.")

    unique_groups = df_distances["group"].unique()
    num_groups = len(unique_groups)

    # determine the maximum y-axis limits for synchronizing across all pages
    max_distance = max(
        df_distances["best_matching_distance_genotype"].max(),
        df_distances["best_matching_distance_overall"].max(),
    )
    ylim_distances = max_distance * 1.03
    if df_hit_counts is not None:
        max_hit_count = df_hit_counts["num_of_hits_total"].max()
        ylim_hit_counts = max_hit_count * 1.03
    else:
        ylim_hit_counts = None

    num_pages = math.ceil(num_groups / max_groups_per_page)

    for page in range(num_pages):
        log.info(f"Plotting page {page + 1} of {num_pages} ...")

        groups = set(unique_groups[page * max_groups_per_page : (page + 1) * max_groups_per_page])
        df_page = df_distances[df_distances["group"].isin(groups)].copy()

        plotter = MotifDistancePlotter(
            df_page,
            df_hit_counts=df_hit_counts,
            allele_column=allele_column,
            alleles_in_reference=alleles_in_reference,
            used_pct_rank_threshold=used_pct_rank_threshold,
            ylim_distances=ylim_distances,
            ylim_hit_counts=ylim_hit_counts,
        )
        yield plotter.plot()


def _sort_groups_by_motif_distance(
    df: pd.DataFrame,
    mode: str,
    metric: str,
    ascending: bool,
) -> pd.DataFrame:
    """Sort the groups by the motif distance.

    This function sorts the groups in the DataFrame by the specified metric (either the median
    or mean of the best matching distances) and mode (either ratio, genotype, or overall).

    Args:
        df: DataFrame with the distances to the best matching motifs.
        mode: The mode to sort by, either 'ratio', 'genotype', or 'overall'.
        metric: The metric to use for sorting, either 'median' or 'mean'.
        ascending: Whether to sort in ascending order.

    Returns:
        DataFrame sorted by the specified metric and mode.

    Raises:
        ValueError: If the mode or metric is not valid.
    """
    if mode not in ("ratio", "genotype", "overall"):
        raise ValueError("mode must be either 'ratio', 'genotype', or 'overall'")

    if metric not in ("median", "mean"):
        raise ValueError("metric must be either 'median' or 'mean'")

    df_grouped = df.groupby("group").agg(
        {
            "best_matching_distance_genotype": metric,
            "best_matching_distance_overall": metric,
        }
    )

    if mode == "ratio":
        df_grouped["sorting_metric"] = df_grouped["best_matching_distance_genotype"] / df_grouped[
            "best_matching_distance_overall"
        ].apply(
            lambda x: max(x, EPSILON)  # avoid division by zero or very small values
        )
    elif mode == "genotype":
        df_grouped["sorting_metric"] = df_grouped["best_matching_distance_genotype"]
    else:
        df_grouped["sorting_metric"] = df_grouped["best_matching_distance_overall"]

    group2metric = df_grouped["sorting_metric"].to_dict()

    df = (
        df.assign(temp_sorting_metric=df["group"].map(group2metric))
        .sort_values("temp_sorting_metric", ascending=ascending, kind="stable")
        .drop(columns=["temp_sorting_metric"])
        .reset_index(drop=True)
    )

    return df
