"""Define callbacks for click arguments."""
from __future__ import annotations

from pathlib import Path

import click
from cloudpathlib import CloudPath


def abort_if_not_exists(
    ctx: click.Context,  # noqa: U100
    param: click.Option | click.Parameter,  # noqa: U100
    value: Path | CloudPath | None,
) -> Path | CloudPath:
    """Ensure the parameter value corresponds to an existing directory/file.

    Args:
        ctx: click context. It is required by click.
        param: click parameter object. It is required by click.
        value: value associated to the parameter, it must be of type pathlib.Path or
            cloudpathlib.CloudPath

    Returns:
        local/remote path

    Raises:
        click.BadParameter: if the local/remote path does not exist
    """
    if value is not None and not value.exists():
        raise click.BadParameter(f"The provided path {value} does not exist.")

    return value
