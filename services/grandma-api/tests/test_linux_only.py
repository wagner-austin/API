"""The service runs on Linux only, and the tree can prove it.

Until 2026-10-21 (MCPs board task 703fec5a) ``make up`` and ``make down``
ran a Windows host mode, ``scripts/GrandmaService.ps1`` with ``start.ps1``
and ``stop.ps1``, behind a Windows fence in the Makefile, and a Pester suite
under ``tests/pester`` tested it. The operator ruled "well covnert any
windows code to linux no back compat", and the port deleted all of it: up
and down now drive this directory's compose file on any Linux host.

A deleted host mode comes back the way it arrived: one ``.ps1`` beside a
recipe, one ``ifeq ($(OS),Windows_NT)`` arm, each passed by every other
test. These read the package for exactly those shapes, and each detector is
shown finding a synthetic one first, so a clean package is a measurement
and not a detector that cannot fire. These read what the package carries;
that the recipes then RUN on Linux is no file read's to say.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

_PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parents[1]

#: PowerShell's script, module and manifest suffixes, compared case-blind.
_POWERSHELL_SUFFIXES: Final[frozenset[str]] = frozenset({".ps1", ".psm1", ".psd1"})

#: Directories a package manager fills with third-party files, never this
#: package's own: poetry's in-project venv and the web frontend's npm tree.
_INSTALLED_DIRECTORIES: Final[frozenset[str]] = frozenset({".venv", "node_modules"})

#: A Makefile's Windows tells, in the lines make runs or assigns, compared
#: case-blind.
_MAKEFILE_TELLS: Final[tuple[str, ...]] = ("$(os)", "windows_nt", "powershell", "pwsh", ".ps1")


def powershell_files(root: Path) -> list[str]:
    """Return every PowerShell file under root, as a POSIX path relative to it.

    Args:
        root: The directory to walk.

    Returns:
        Each file whose suffix is a PowerShell one, sorted, with the
        installed directories' contents left out.
    """
    found: list[str] = []
    for directory, subdirectories, files in os.walk(root):
        subdirectories[:] = [name for name in subdirectories if name not in _INSTALLED_DIRECTORIES]
        found.extend(
            (Path(directory) / name).relative_to(root).as_posix()
            for name in files
            if Path(name).suffix.lower() in _POWERSHELL_SUFFIXES
        )
    return sorted(found)


def makefile_tells(text: str) -> list[str]:
    """Return each live Makefile line carrying a Windows tell.

    Args:
        text: The Makefile's text.

    Returns:
        Every line, comments excluded, holding one of the tells.
    """
    return [
        line
        for line in text.splitlines()
        if not line.lstrip().startswith("#")
        and any(tell in line.lower() for tell in _MAKEFILE_TELLS)
    ]


def test_the_powershell_detector_finds_each_shape(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "GrandmaService.ps1").write_text("param()\n", encoding="utf-8")
    (tmp_path / "tests" / "pester").mkdir(parents=True)
    (tmp_path / "tests" / "pester" / "Helpers.PSM1").write_text("\n", encoding="utf-8")
    (tmp_path / "web" / "node_modules" / ".bin").mkdir(parents=True)
    (tmp_path / "web" / "node_modules" / ".bin" / "vitest.ps1").write_text("\n", encoding="utf-8")
    (tmp_path / "start.ps1.md").write_text("\n", encoding="utf-8")
    assert powershell_files(tmp_path) == [
        "scripts/GrandmaService.ps1",
        "tests/pester/Helpers.PSM1",
    ]


def test_the_package_carries_no_powershell() -> None:
    assert (_PROJECT_ROOT / "Makefile").is_file()
    assert powershell_files(_PROJECT_ROOT) == []
    assert not (_PROJECT_ROOT / "tests" / "pester").exists()


def test_the_makefile_detector_finds_each_tell() -> None:
    text = (
        "# Windows_NT was the old fence\n"
        "ifeq ($(OS),Windows_NT)\n"
        "SHELL := PowerShell.exe\n"
        "\tpwsh -File scripts/start.ps1\n"
        "\t& scripts/stop.ps1\n"
        "up:\n"
    )
    assert makefile_tells(text) == [
        "ifeq ($(OS),Windows_NT)",
        "SHELL := PowerShell.exe",
        "\tpwsh -File scripts/start.ps1",
        "\t& scripts/stop.ps1",
    ]


def test_the_makefile_has_no_platform_fence() -> None:
    text = (_PROJECT_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "\nup:\n" in text
    assert "\ndown:\n" in text
    assert makefile_tells(text) == []
