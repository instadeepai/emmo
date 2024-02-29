"""Module for functions to plot motifs."""
from __future__ import annotations

import logomaker as lm
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from emmo.constants import NATURAL_AAS
from emmo.models.cleavage import CleavageModel
from emmo.models.deconvolution import DeconvolutionModelMHC2
from emmo.pipeline.background import Background
from emmo.pipeline.background import BackgroundType


LOGO_PROPS = {"fade_below": 0.5, "shade_below": 0.5}


def plot_single_ppm(
    ppm: np.ndarray,
    alphabet: list[str] | tuple[str, ...] | None = None,
    background: BackgroundType | None = None,
    ax: matplotlib.axes._axes.Axes | None = None,
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
    axs: matplotlib.axes._axes.Axes | np.ndarray[matplotlib.axes._axes.Axes] | None = None,
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

    _plot_mhc2_model_to_axes(model, title, axs)
    plt.tight_layout()
    plt.show()


def _plot_mhc2_model_to_axes(
    model: DeconvolutionModelMHC2,
    title: str,
    axs: matplotlib.axes._axes.Axes | np.ndarray,
) -> None:
    """Plot an MHC2 deconvolution model using the given Axes instance(s).

    Args:
        model: The model.
        title: The title for the plot.
        axs: The Axes instance(s) used for plotting.

    Raises:
        RuntimeError: If the provided Axes array is not large enough.
    """
    cum_class_weights = np.sum(model.class_weights, axis=1)
    number_of_classes = model.number_of_classes

    for i in range(number_of_classes):
        try:
            ax = axs[i]
        except TypeError:
            if i == 0:
                ax = axs
            else:
                raise RuntimeError(
                    "invalid or not enough axes provides for model with {number_of_classes} classes"
                )

        plot_single_ppm(
            model.ppm[i],
            alphabet=tuple(model.alphabet),
            background=model.background,
            ax=ax,
        )

        title_to_write = f"motif {i+1} of {number_of_classes} (weight {cum_class_weights[i]:.3f})"

        if title:
            title_to_write = f"{title}\n{title_to_write}"

        ax.set_title(title_to_write)


def plot_cleavage_model(model: CleavageModel) -> None:
    """Plot a cleavage model.

    Args:
        model: The cleavage model.
    """
    number_of_classes = model.number_of_classes

    cum_class_weights_n = model.class_weights_n
    cum_class_weights_c = model.class_weights_c

    _, axs = plt.subplots(2, number_of_classes, figsize=(4 * number_of_classes, 8))

    for i, name, ppm, cum_class_weights in zip(
        range(2),
        ["N terminus", "C terminus"],
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

    plt.tight_layout()
    plt.show()
