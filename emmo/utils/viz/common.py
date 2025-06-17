"""Module to define common functions used for all the visualisations."""
from __future__ import annotations

import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
from cloudpathlib import AnyPath
from cloudpathlib import CloudPath
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


def save_plot(file_path: Openable, figures: list[Figure] | None = None) -> None:
    """Save the current plot locally or remotely and close it.

    Args:
        file_path: Local or remote path where to save the plot.
        figures: List of Figure instances to save.
    """
    file_path = AnyPath(file_path)

    if isinstance(file_path, CloudPath):
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_file = Path(tmp_dir) / file_path.name
            if figures is not None:
                with PdfPages(tmp_file) as pdf:
                    for fig in figures:
                        pdf.savefig(fig)
            else:
                plt.savefig(tmp_file)
            upload_to_directory(tmp_file, file_path.parent, force=True)
    else:
        file_path.parent.mkdir(parents=True, exist_ok=True)
        if figures is not None:
            with PdfPages(file_path) as pdf:
                for fig in figures:
                    pdf.savefig(fig)
        else:
            plt.savefig(file_path)

    log.info(f"Saved plot at {file_path}")

    plt.close()
