"""This package tracks no Windows script (board task e705f08c).

Until 2026-10-25 tools/fleet tracked 38 PowerShell files: the committed
renders of its Windows dialect under ``rendered/``, the Windows install
scripts under ``scripts/`` and the Pester suites under ``tests/pester/``.
The operator ruled that day "so no windows shit then ok", loki, the last
Windows node, left the roster, and every one of them was deleted. This reads
what git tracks under the package, as the staged tree on a fleet node does
too, and fails naming any ``ps1``, ``psm1``, ``psd1``, ``bat`` or ``cmd``
file, so a Windows script that comes back fails here before it ships.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

#: This package's directory: tools/fleet/tests is one level below it.
PACKAGE: Final = Path(__file__).resolve().parents[1]

#: The suffixes of a script only Windows runs, compared without case.
WINDOWS_SUFFIXES: Final[tuple[str, ...]] = (".ps1", ".psm1", ".psd1", ".bat", ".cmd")


def windows_scripts(paths: tuple[str, ...]) -> list[str]:
    """The paths that name a Windows script.

    Args:
        paths: Tracked paths.

    Returns:
        Each one whose suffix is a Windows script's, in order.
    """
    return [path for path in paths if Path(path).suffix.lower() in WINDOWS_SUFFIXES]


def _tracked() -> tuple[str, ...]:
    """Every path git tracks under this package.

    Returns:
        The paths, relative to the package.
    """
    done = subprocess.run(
        ["git", "ls-files", "-z", "--", "."],
        cwd=PACKAGE,
        capture_output=True,
        text=True,
        check=True,
        timeout=120,
    )
    return tuple(entry for entry in done.stdout.split("\0") if entry)


def test_the_package_tracks_no_windows_script() -> None:
    tracked = _tracked()

    assert windows_scripts(tracked) == []
    # Not a pass over nothing: the package tracked 552 files on 2026-10-25.
    assert len(tracked) > 500, f"git tracks {len(tracked)} file(s) here; the package ships more"


def test_every_windows_suffix_is_named_whatever_its_case() -> None:
    paths = (
        "rendered/dialect-build.ps1",
        "scripts/Install.PSM1",
        "scripts/module.psd1",
        "run.bat",
        "run.CMD",
        "src/fleet/core/dialect.py",
        "docs/ps1-notes.md",
        "tests/cmd_line.py",
    )

    assert windows_scripts(paths) == [
        "rendered/dialect-build.ps1",
        "scripts/Install.PSM1",
        "scripts/module.psd1",
        "run.bat",
        "run.CMD",
    ]
