"""Run a package's guard bootstrap when it has one.

Every ``make lint`` used to carry a PowerShell conditional: if
``scripts/guard.py`` or ``scripts/guard/__main__.py`` exists, run
``poetry run python -m scripts.guard`` and stop on its exit status. This is
that conditional, portable, and it SAYS when it ran nothing: a package with
no guard shim prints that it examined zero rules rather than exiting 0 in
silence, which is the difference between "clean" and "not checked".

``python -m scripts.guard`` from the package directory is the single form
every Makefile uses; running the file by path puts ``scripts/`` on
``sys.path[0]`` rather than the package root, which is a different program.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

from maketools import _test_hooks

#: The two shapes a guard bootstrap takes.
GUARD_SHIMS: Final[tuple[str, ...]] = ("scripts/guard.py", "scripts/guard/__main__.py")

#: How the shim is invoked.
GUARD_ARGV: Final[tuple[str, ...]] = ("poetry", "run", "python", "-m", "scripts.guard")


def run_guard(project: Path) -> int:
    """Run the guard shim, or say there is none.

    Args:
        project: The package directory.

    Returns:
        The shim's exit status, or 0 with a "not applicable" line when the
        package has no shim.
    """
    present = [shim for shim in GUARD_SHIMS if (project / shim).is_file()]
    if not present:
        _test_hooks.write_line(
            f"guard: not applicable: {project.name} has no scripts/guard.py or "
            "scripts/guard/__main__.py, 0 rules run"
        )
        return 0
    _test_hooks.write_line(f"guard: running {present[0]}")
    return _test_hooks.run_inheriting(
        GUARD_ARGV, cwd=project, env=_test_hooks.environ(), new_session=False
    )


__all__ = ["GUARD_ARGV", "GUARD_SHIMS", "run_guard"]
