"""Run MCPs' PowerShell harness over this repository at the commit API pins.

MCPs board task 939ec5c7. ``tools/ps-harness``'s recipe, which the root's
``check-powershell``, API's powershell workflow and the fleet project of the
same name all run, passed the pin as ``--ref $(file <...mcps-harness.sha)``.
``$(file ...)`` exists from GNU make 4.0, and serendipity's make is GnuWin32
make 3.81, where it reads as nothing: fleet job 2d949933 at 1f767a97f
(repo API) ran ``ps-harness --ref  ../../../MCPs ../..`` and was refused
``MAKETOOLS_USAGE``. The grammar bans ``$(shell ...)``
(:mod:`maketools.makefile_grammar`), so make has no portable way to read a
file, and the pin is read here instead, under every make.

The pin is passed as read, stripped of its line ending. It is not validated
here: the command it reaches refuses an empty ref as usage and a commit MCPs
does not carry as ``PS_HARNESS_UNPUBLISHED``, each by name, and a second
check here would be a restatement of those.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Final

from maketools import _test_hooks

#: The file holding the MCPs commit whose harness API runs, beside the hooks.
PIN_PATH: Final[Path] = Path(".githooks", "mcps-harness.sha")

#: The script that runs one MCPs maketools command as MCPs published it.
PUBLISHED_MAKETOOLS: Final[Path] = Path(".githooks", "published_maketools.py")

#: Wall clock on the harness. One hour: the whole run over API measured 353 s
#: on the hub on 2026-10-03 (198 Pester cases over 44 scripts), and the fleet
#: project expects 25 minutes, so this clears both and is still FINITE.
HARNESS_WALL_SECONDS: Final[int] = 3600


def harness_argv(repo_root: Path, pin: str) -> tuple[str, ...]:
    """The command that runs the published harness at ``pin`` over ``repo_root``.

    Args:
        repo_root: This repository's root, which MCPs sits beside.
        pin: The MCPs commit, as the pin file holds it less its line ending.

    Returns:
        This interpreter, the published-maketools script and its
        ``ps-harness --ref <pin> <MCPs> <repo_root>`` arguments, MCPs being
        the directory beside the repository where the script itself looks.
    """
    return (
        sys.executable,
        str(repo_root / PUBLISHED_MAKETOOLS),
        "ps-harness",
        "--ref",
        pin,
        str(repo_root.parent / "MCPs"),
        str(repo_root),
    )


def run_pinned_harness(repo_root: Path) -> int:
    """Read the pin and run the harness at it.

    Args:
        repo_root: This repository's root.

    Returns:
        The harness's exit status.

    Raises:
        FileNotFoundError: When the pin file is absent, a repository that
            has lost the file the harness is defined by.
        subprocess.TimeoutExpired: When the harness outlives
            :data:`HARNESS_WALL_SECONDS`.
    """
    pin = (repo_root / PIN_PATH).read_text(encoding="utf-8").strip()
    return _test_hooks.run_inheriting(
        harness_argv(repo_root, pin),
        cwd=repo_root,
        env=_test_hooks.environ(),
        new_session=False,
        timeout_seconds=HARNESS_WALL_SECONDS,
    )


__all__ = [
    "HARNESS_WALL_SECONDS",
    "PIN_PATH",
    "PUBLISHED_MAKETOOLS",
    "harness_argv",
    "run_pinned_harness",
]
