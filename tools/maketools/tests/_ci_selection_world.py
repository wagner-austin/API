"""A scratch repository CI's package selection runs in, for the cases that run it.

``scripts/ci_select_packages.py`` finds its repository from its own path, so a
copy of it placed in a scratch repository selects among that repository's
packages by that repository's declaration. The wiring cases and the inputs
cases both build one here.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final

#: The repository root: tools/maketools/tests/ is three levels below it.
ROOT: Final[Path] = Path(__file__).resolve().parents[3]


def ci_repository(repo: Path, declaration: str, packages: Mapping[str, Sequence[str]]) -> Path:
    """Lay out a repository holding CI's selection and gated packages.

    Args:
        repo: The scratch directory.
        declaration: The text of its ``scripts/make/project-roots.json``.
        packages: Each package directory and the package directories it takes
            as Poetry path dependencies.

    Returns:
        The copy of ``scripts/ci_select_packages.py`` inside it.
    """
    (repo / "scripts" / "make").mkdir(parents=True, exist_ok=True)
    script = repo / "scripts" / "ci_select_packages.py"
    shutil.copyfile(ROOT / "scripts" / "ci_select_packages.py", script)
    (repo / "scripts" / "make" / "project-roots.json").write_text(declaration, encoding="utf-8")
    for package, dependencies in packages.items():
        (repo / package).mkdir(parents=True, exist_ok=True)
        (repo / package / "Makefile").write_text("check:\n\ttrue\n", encoding="utf-8")
        depth = "/".join(".." for _ in package.split("/"))
        lines = "".join(
            f'{dependency.rsplit("/", 1)[-1]} = {{ path = "{depth}/{dependency}" }}\n'
            for dependency in dependencies
        )
        manifest = f"[tool.poetry.dependencies]\n{lines}"
        (repo / package / "pyproject.toml").write_text(manifest, encoding="utf-8")
    return script


def run_selection(script: Path, arguments: Sequence[str]) -> subprocess.CompletedProcess[str]:
    """Run a copy of CI's selection as the workflow runs it.

    Args:
        script: The copy :func:`ci_repository` placed.
        arguments: Its arguments.

    Returns:
        The finished run, its output captured.
    """
    return subprocess.run(
        [sys.executable, str(script), *arguments], capture_output=True, text=True, check=False
    )
