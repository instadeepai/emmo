"""Module defines all available cli tools."""
from __future__ import annotations

import click

from emmo.cli.binding_prediction import predict_mhc2
from emmo.cli.bucket import pull_model
from emmo.cli.bucket import push_model
from emmo.cli.deconvolution import deconvolute_mhc2


@click.group()
def main() -> None:
    """Entry point for emmo."""
    pass


main.add_command(deconvolute_mhc2)
main.add_command(predict_mhc2)

main.add_command(pull_model)
main.add_command(push_model)


if __name__ == "__main__":
    main()
