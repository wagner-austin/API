"""CLI: write every rendered JSON document to its committed copy.

Usage:
    python -m fleet.cli.render_documents --out rendered

``make render-documents`` runs exactly that. It writes the hooks check pins
(:mod:`fleet.core.hooks_check_document`) and the toolchain's pinned versions
(:mod:`fleet.core.toolchain_document`); why each exists and its shape are
those modules'. ``tests/test_hooks_check_document.py`` and
``tests/test_toolchain_document.py`` fail until each committed copy is its
render, so a pin moved without running this is never published stale.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Callable, Sequence
from typing import Final

from platform_core import cli_args
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging

from fleet.core import _test_hooks
from fleet.core.hooks_check_document import HOOKS_CHECK_DOCUMENT, render_hooks_check_document
from fleet.core.toolchain_document import TOOLCHAIN_DOCUMENT, render_toolchain_document

_log = get_logger(__name__)

OUT_FLAG = "--out"

_FLAGS = (OUT_FLAG,)

#: Every rendered document: its file name under ``--out`` and its renderer.
DOCUMENTS: Final[tuple[tuple[str, Callable[[], str]], ...]] = (
    (HOOKS_CHECK_DOCUMENT, render_hooks_check_document),
    (TOOLCHAIN_DOCUMENT, render_toolchain_document),
)


def main(argv: Sequence[str] | None = None) -> int:
    """Write the documents.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once every document is written.

    Raises:
        ValueError: When a flag is unknown, repeated or missing its value.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    for name, render in DOCUMENTS:
        _test_hooks.write_text(out / name, render())
        _log.info("wrote %s to %s", name, out)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-render-documents",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(main())


__all__ = ["DOCUMENTS", "OUT_FLAG", "entrypoint", "main"]


# Without this, `python -m fleet.cli.render_documents` imports the module,
# writes nothing and exits 0, and a stale copy would read as current.
if __name__ == "__main__":
    entrypoint()
