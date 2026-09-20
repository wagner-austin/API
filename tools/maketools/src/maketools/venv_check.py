"""Remove a package's ``.venv`` when it can no longer answer.

Every ``make lint`` used to open with a PowerShell line that asked
``poetry run mypy --version`` and removed ``.venv`` when the answer was
not zero, so the ``poetry sync`` that follows builds a fresh one instead of
failing inside a venv whose interpreter moved or whose packages half
installed. This is that line, portable.

A MISSING ``.venv`` IS NOT STALE: ``poetry run`` fails there too, but there
is nothing to remove and the sync creates it. Removing is only for a venv
that exists and does not work.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from maketools import _test_hooks

#: The question that decides staleness; mypy is the first tool lint runs.
PROBE_ARGV: Final[tuple[str, ...]] = ("poetry", "run", "mypy", "--version")

#: The directory poetry keeps the environment in, with in-project venvs.
VENV_DIRECTORY: Final[str] = ".venv"


def check_venv(project: Path) -> bool:
    """Probe the venv and remove it when it cannot answer.

    Args:
        project: The package directory.

    Returns:
        True when the venv was removed.
    """
    venv = project / VENV_DIRECTORY
    if not venv.is_dir():
        _test_hooks.write_line("venv-check: no .venv yet; poetry sync will create one")
        return False
    result = _test_hooks.run_capturing(PROBE_ARGV, cwd=project)
    if result["returncode"] == 0:
        _test_hooks.write_line(f"venv-check: .venv answers ({result['stdout'].strip()})")
        return False
    _test_hooks.write_line(
        f"venv-check: stale venv detected (poetry run mypy --version exited "
        f"{result['returncode']}); removing {venv}"
    )
    _test_hooks.remove_tree(venv)
    return True


__all__ = ["PROBE_ARGV", "VENV_DIRECTORY", "check_venv"]
