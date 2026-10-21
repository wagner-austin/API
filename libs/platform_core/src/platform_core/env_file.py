"""The compose stack's ``.env``, read by one parser in this monorepo.

WHOSE FILE, WHOSE GRAMMAR. The MCPs repository's ``.env`` holds the
stack's credentials: ``MCP_INTERNAL_KEY`` and ``OPERATOR_TENANT_ID`` for the
board, ``TAILNET_WRITER_CLIENT_ID`` and ``TAILNET_WRITER_CLIENT_SECRET`` for
the netboot arm. MCPs owns the file, and
its reader, ``packages/maketools`` ``with_env.read_env_file``, states the
grammar: blank lines and ``#`` comments are skipped; every other line is
``KEY=VALUE``, split at the first ``=`` with the key stripped and the value
kept raw, not unquoted; the last assignment of a key wins; and a line with
no ``=`` is refused rather than skipped, since skipping it would silently
drop whichever key it was setting. This module reads it the same way, so
the two repositories cannot disagree about one file.

ONE PARSER, LIFTED (MCPs board task af8b5662). Until then this monorepo
held two readers of that one file with two grammars:
:func:`platform_core.session_label.stack_credentials` unquoted values and
skipped a line it could not split, and the fleet's netboot arm read one
``KEY=`` prefix and refused a repeated key. Measured on austinpc and
diphtheria on 2026-10-19, the live files carry no quoted value, no repeated
key and no line without ``=``, so this grammar reads them exactly as both
readers did.

A refusal names the source and the line NUMBER and never repeats the line,
because a mis-typed credential is exactly the line that would be echoed into
a log. The file's bytes are the caller's to read, through its own seam, as
with :mod:`platform_core.env_assignments`, the parser of the PowerShell
credentials file, whose grammar is a different one.
"""

from __future__ import annotations

from collections.abc import Mapping

from platform_core.errors import AppError, ErrorCode


def parse_env_file(text: str, *, source: str) -> Mapping[str, str]:
    """Parse a ``KEY=VALUE`` env file in the grammar MCPs' reader holds.

    Args:
        text: The file's contents, already decoded.
        source: What the text was read from, for the refusal message.

    Returns:
        Each key's value, the last assignment winning.

    Raises:
        AppError: ``CONFIG_ERROR`` naming the source and line number for a
            non-blank, uncommented line with no ``=``.
    """
    values: dict[str, str] = {}
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if line == "" or line.startswith("#"):
            continue
        key, separator, value = line.partition("=")
        if separator == "":
            raise AppError(
                ErrorCode.CONFIG_ERROR,
                f"line {number} of {source} is not a KEY=VALUE line; only those, blank lines "
                "and # comments are read, and skipping one would drop whichever key it sets",
            )
        values[key.strip()] = value
    return values


__all__ = ["parse_env_file"]
