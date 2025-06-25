"""Module to define common functions used for all the visualisations."""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Iterator

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from cloudpathlib import AnyPath
from cloudpathlib import CloudPath
from matplotlib.axes import Axes
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

from emmo.bucket.io import upload_to_directory
from emmo.io.file import Openable
from emmo.utils import logger

log = logger.get(__name__)


LOGO_PROPS = {"fade_below": 0.5, "shade_below": 0.5}
COLOR_DEFAULT = "royalblue"
COLOR_DEFAULT_LIGHT = "skyblue"
COLOR_ALT = "sienna"
COLOR_ALT_LIGHT = "bisque"


def save_plot(
    file_path: Openable,
    figures: list[Figure] | Iterator[Figure] | None = None,
    close_figures: bool = True,
) -> None:
    """Save the current plot locally or remotely.

    If `figures` is provided, save each figure in a PDF file. Otherwise, save the current active
    plot. By default, the figures will be closed after saving.

    Args:
        file_path: Local or remote path where to save the plot.
        figures: Optional iterator of matplotlib Figure objects to save.
        close_figures: Whether to close the figures after saving them.
    """
    file_path = AnyPath(file_path)

    if isinstance(file_path, CloudPath):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / file_path.name
            _save_pdf_local(tmp_file, figures, close_figures)
            upload_to_directory(tmp_file, file_path.parent, force=True)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        _save_pdf_local(file_path, figures, close_figures)

    log.info(f"Saved plot at {file_path}")


def unify_ylim(axs: np.ndarray[Axes] | list[Axes]) -> None:
    """Unify the the ylim of Axes objects to the minimum and maximum.

    Args:
        axs: Array of Axes instances.
    """
    axs = axs.flatten() if isinstance(axs, np.ndarray) else axs
    min_ylim = min(ax.get_ylim()[0] for ax in axs)
    max_ylim = max(ax.get_ylim()[1] for ax in axs)
    for ax in axs:
        ax.set_ylim(min_ylim, max_ylim)


def rectangle_with_text(
    x: float,
    y: float,
    width: float,
    height: float,
    ax: plt.Axes,
    facecolor: str = "lightgrey",
    text: str | None = None,
    fontsize: int = 10,
    text_color: str = "black",
    text_rotation: int | float = 0,
) -> None:
    """Plot a rectangle with optional text inside it.

    Args:
        x: X-coordinate of the rectangle's bottom-left corner.
        y: Y-coordinate of the rectangle's bottom-left corner.
        width: Width of the rectangle.
        height: Height of the rectangle.
        ax: Matplotlib Axes object to plot on.
        facecolor: Color of the rectangle's face.
        text: Optional text to display inside the rectangle.
        fontsize: Font size for the text.
        text_color: Color of the text.
        text_rotation: Rotation angle for the text in degrees.
    """
    rectangle = patches.Rectangle(
        (x, y),
        width,
        height,
        linewidth=0.8,
        edgecolor="white",
        facecolor=facecolor,
    )
    ax.add_patch(rectangle)

    if text is not None:
        ax.text(
            x + width / 2 + 0.03,  # adjusted for better centering
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=text_color,
            rotation=text_rotation,
        )


def _save_pdf_local(
    file_path: Openable,
    figures: list[Figure] | Iterator[Figure] | None,
    close_figures: bool,
) -> None:
    """Save figures to a PDF file at a given local path.

    Args:
        file_path: Path to the PDF file where figures will be saved.
        figures: Optional iterator of matplotlib Figure objects to save.
        close_figures: Whether to close the figures after saving them.
    """
    file_path = AnyPath(file_path)

    if figures is not None:
        with PdfPages(file_path) as pdf:
            for fig in figures:
                pdf.savefig(fig)
                if close_figures:
                    fig.clear()
                    plt.close(fig)
    else:
        # save the current active plot
        plt.savefig(file_path)
        if close_figures:
            plt.close()


def is_axes_empty(ax: Axes) -> bool:
    """Check if a Matplotlib Axes object is empty.

    An Axes is considered empty if it contains no lines, patches, texts, collections, images,
    artists, or containers.

    Args:
        ax: Matplotlib Axes object to check.

    Returns:
        True if the Axes is empty, False otherwise.
    """
    return (
        len(ax.lines) == 0
        and len(ax.patches) == 0
        and len(ax.texts) == 0
        and len(ax.collections) == 0
        and len(ax.images) == 0
        and len(ax.artists) == 0
        and len(ax.containers) == 0
    )
