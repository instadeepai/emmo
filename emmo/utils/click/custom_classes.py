"""Define custom classes for click-related functionalities."""
from __future__ import annotations

import click


class FullHelpGroup(click.Group):
    """Custom class to show the full description of subcommands of a group."""

    def format_commands(
        self, ctx: click.core.Context, formatter: click.formatting.HelpFormatter
    ) -> None:
        """Format the commands list for the help message.

        Args:
            ctx: The context for the current command.
            formatter: The formatter to write the help message to.
        """
        rows = []

        for subcommand in self.list_commands(ctx):
            cmd = self.get_command(ctx, subcommand)
            if cmd is None:
                continue

            help_text = cmd.get_short_help_str(limit=45)
            full_description = cmd.help if cmd.help else help_text
            if "\n" in full_description:
                full_description = full_description.split("\n")[0]

            rows.append((subcommand, full_description))

        if rows:
            with formatter.section("Commands"):
                formatter.write_dl(rows)
