"""CLI for the bot runs index (``runs/bot/_index.tsv``).

Three subcommands, each returning a CLI exit code:

  - ``bot-runs list`` -- print every row, newest last.
  - ``bot-runs find <pattern>`` -- print rows whose ``stamp`` or
    ``exit_reason`` contains ``pattern`` (case-insensitive substring).
  - ``bot-runs show <stamp>`` -- print one row in long, label/value form.

Output is written to ``sys.stdout``; errors go to ``sys.stderr`` and
the process exits with status ``1``. The CLI uses
:mod:`tankpit_bot.diagnostics.runs_index` for all I/O so the same
read paths are exercised by both production and the CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path

from tankpit_bot.diagnostics.runs_index import (
    DEFAULT_INDEX_PATH,
    INDEX_COLUMNS,
    BotRunIndexRowDict,
    find_row,
    load_index_rows,
)

_USAGE_BLOCK = (
    "usage: bot-runs <list | find PATTERN | show STAMP>\n"
    "  list           Print every row in the index.\n"
    "  find PATTERN   Print rows whose stamp/exit_reason contains PATTERN.\n"
    "  show STAMP     Print one row by exact stamp match.\n"
)


def _format_row_for_list(row: BotRunIndexRowDict) -> str:
    """Format one row as a TSV-aligned summary line.

    Args:
        row: Row to format.

    Returns:
        Single line (no trailing newline) with columns separated by
        tabs in :data:`INDEX_COLUMNS` order.
    """
    return "\t".join(
        [
            row["stamp"],
            str(row["duration_s"]),
            row["exit_reason"],
            str(row["ticks"]),
            str(row["stalls"]),
            str(row["shots_fired"]),
            str(row["kills"]),
            f"{row['kills_per_min']:.2f}",
        ]
    )


def _format_row_for_show(row: BotRunIndexRowDict) -> str:
    """Format one row as a long label/value block.

    Args:
        row: Row to format.

    Returns:
        Multiline block with one column per line.
    """
    lines = [
        f"stamp:          {row['stamp']}",
        f"duration_s:     {row['duration_s']}",
        f"exit_reason:    {row['exit_reason']}",
        f"ticks:          {row['ticks']}",
        f"stalls:         {row['stalls']}",
        f"shots_fired:    {row['shots_fired']}",
        f"kills:          {row['kills']}",
        f"kills_per_min:  {row['kills_per_min']:.2f}",
    ]
    return "\n".join(lines) + "\n"


def run_list(index_path: Path = DEFAULT_INDEX_PATH) -> int:
    """List every row in the index.

    Args:
        index_path: Index path. Defaults to
            :data:`runs_index.DEFAULT_INDEX_PATH`.

    Returns:
        ``0`` on success (even when the index is empty).
    """
    rows = load_index_rows(index_path)
    sys.stdout.write("\t".join(INDEX_COLUMNS) + "\n")
    for row in rows:
        sys.stdout.write(_format_row_for_list(row) + "\n")
    if not rows:
        sys.stdout.write("(no runs recorded)\n")
    return 0


def run_find(pattern: str, index_path: Path = DEFAULT_INDEX_PATH) -> int:
    """Print rows whose ``stamp`` or ``exit_reason`` contains ``pattern``.

    Args:
        pattern: Case-insensitive substring to match.
        index_path: Index path. Defaults to
            :data:`runs_index.DEFAULT_INDEX_PATH`.

    Returns:
        ``0`` when at least one row matched, ``1`` when none did.
    """
    needle = pattern.lower()
    rows = [
        row
        for row in load_index_rows(index_path)
        if needle in row["stamp"].lower() or needle in row["exit_reason"].lower()
    ]
    sys.stdout.write("\t".join(INDEX_COLUMNS) + "\n")
    for row in rows:
        sys.stdout.write(_format_row_for_list(row) + "\n")
    if not rows:
        sys.stderr.write(f"bot-runs: no rows matched {pattern!r}\n")
        return 1
    return 0


def run_show(stamp: str, index_path: Path = DEFAULT_INDEX_PATH) -> int:
    """Print one row in long format.

    Args:
        stamp: Run stamp to look up (exact match).
        index_path: Index path. Defaults to
            :data:`runs_index.DEFAULT_INDEX_PATH`.

    Returns:
        ``0`` on success, ``1`` when no row matches ``stamp``.
    """
    row = find_row(stamp, index_path)
    if row is None:
        sys.stderr.write(f"bot-runs: no row with stamp {stamp!r}\n")
        return 1
    sys.stdout.write(_format_row_for_show(row))
    return 0


def run(argv: list[str]) -> int:
    """Dispatch to a subcommand based on ``argv``.

    Args:
        argv: Argument vector excluding the program name (i.e.
            ``sys.argv[1:]``).

    Returns:
        Subcommand exit code: ``0`` on success, ``1`` on usage error
        or empty result.
    """
    if len(argv) == 0:
        sys.stderr.write(_USAGE_BLOCK)
        return 1
    command = argv[0]
    if command == "list" and len(argv) == 1:
        return run_list()
    if command == "find" and len(argv) == 2:
        return run_find(argv[1])
    if command == "show" and len(argv) == 2:
        return run_show(argv[1])
    sys.stderr.write(_USAGE_BLOCK)
    return 1


def main() -> None:
    """Entry point for the ``tankpit-bot-runs`` console script."""
    sys.exit(run(sys.argv[1:]))


if __name__ == "__main__":
    main()
