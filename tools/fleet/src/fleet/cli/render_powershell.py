"""CLI: write every rendered PowerShell script to its committed copy.

Usage:
    python -m fleet.cli.render_powershell --out rendered

``make render-powershell`` runs exactly that. The registry and the reason the
copies are committed are :mod:`fleet.core.rendered_powershell`'s; this only
writes them. It removes nothing: a copy whose entry is gone is named by
``tests/test_rendered_powershell.py``, which fails until it is deleted, so a
stale script is never dropped without someone reading which one it was.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core import cli_args
from platform_core.cli_boundary import run_cli
from platform_core.logging import LogFormat, LogLevel, get_logger, setup_logging
from platform_core.report_line import emit_error_line, emit_line

from fleet.core import _test_hooks, rendered_powershell

_log = get_logger(__name__)

OUT_FLAG = "--out"

_FLAGS = (OUT_FLAG,)


def main(argv: Sequence[str] | None = None) -> int:
    """Write every render.

    Args:
        argv: Command-line arguments excluding the program name.

    Returns:
        0 once every file is written.

    Raises:
        ValueError: When a flag is unknown, repeated or missing its value.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    parsed = cli_args.parse_single_flags(tokens, _FLAGS)
    out = pathlib.Path(cli_args.require_flag(parsed, OUT_FLAG))
    scripts = rendered_powershell.render_all()
    for script in scripts:
        _test_hooks.write_text(out / f"{script['name']}.ps1", script["text"])
    _log.info("wrote %d rendered PowerShell script(s) to %s", len(scripts), out)
    return 0


def entrypoint() -> None:
    """Console-script entry point.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    setup_logging(
        level=LogLevel.INFO,
        format_mode=LogFormat.TEXT,
        service_name="fleet-render-powershell",
        instance_id=None,
        extra_fields=None,
    )
    raise SystemExit(run_cli(main, emit=emit_line, emit_error=emit_error_line))


__all__ = ["OUT_FLAG", "entrypoint", "main"]


# Without this, `python -m fleet.cli.render_powershell` imports the module,
# writes nothing and exits 0, and a stale copy would read as current.
if __name__ == "__main__":
    entrypoint()
