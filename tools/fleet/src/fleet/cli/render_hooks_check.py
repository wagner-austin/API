"""CLI: write the hooks check pins' JSON document to its committed copy.

Usage:
    python -m fleet.cli.render_hooks_check --out rendered

``make render-hooks-check`` runs exactly that. Why the document exists and
its shape are :mod:`fleet.core.hooks_check_document`'s; this only writes it.
``tests/test_hooks_check_document.py`` fails until the committed copy is
the render, so a pin moved without running this is never published stale.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging

from fleet.core import _test_hooks
from fleet.core.hooks_check_document import HOOKS_CHECK_DOCUMENT, render_hooks_check_document

_log = get_logger(__name__)

OUT_FLAG = "--out"

_FLAGS = (OUT_FLAG,)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the document.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once the document is written.

    Raises:
        ValueError: When a flag is unknown, repeated or missing its value.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    _test_hooks.write_text(out / HOOKS_CHECK_DOCUMENT, render_hooks_check_document())
    _log.info("wrote %s to %s", HOOKS_CHECK_DOCUMENT, out)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-render-hooks-check",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["OUT_FLAG", "entrypoint", "main"]


# Without this, `python -m fleet.cli.render_hooks_check` imports the module,
# writes nothing and exits 0, and a stale copy would read as current.
if __name__ == "__main__":
    entrypoint()
