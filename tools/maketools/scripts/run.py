#!/usr/bin/env python3
"""The launcher every Makefile recipe calls.

``$(PYTHON) <depth>/tools/maketools/scripts/run.py <command> [args]``, with
the SYSTEM interpreter: no venv, no install, standard library only. The one
thing this file does that ``python -m maketools`` could not is put the
package's ``src`` on the path from wherever the recipe runs, so the same
line works from ``libs/platform_core`` and from ``clients/TankpitBot``
without either knowing where this package is installed -- because it is
not installed.

``libs/platform_core/src`` goes on the path beside it, for the same reason
and by the same route ``scripts/guard.py`` takes to ``monorepo_guards``:
the monorepo permits one error module and one JSON reader, both in
``platform_core``, and the three modules this package imports from it
(``errors``, ``json_utils``, ``error_codes_tooling``) are standard library
all the way down, so they load under the system interpreter.

Everything else is :func:`maketools.cli.main`; this file has no logic of its
own to test apart from the paths it inserts and the exit code it forwards,
and the suite covers both by running it as a real child process.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

PACKAGE_ROOT: Final[Path] = Path(__file__).resolve().parent.parent
SOURCE_ROOTS: Final[tuple[Path, ...]] = (
    PACKAGE_ROOT / "src",
    PACKAGE_ROOT.parent.parent / "libs" / "platform_core" / "src",
)


def main(argv: list[str]) -> int:
    """Insert the source roots on the path and run the CLI.

    Args:
        argv: Arguments after the launcher's own path.

    Returns:
        The CLI's exit code.
    """
    for root in reversed(SOURCE_ROOTS):
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    from maketools.cli import main as cli_main

    return cli_main(argv)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
