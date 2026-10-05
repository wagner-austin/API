"""Execute a rendered build and verify preparation receives its project directory."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from fleet.contracts.source import InstallStep
from fleet.core.dialect_linux import LinuxDialect


@pytest.mark.skipif(sys.platform == "win32", reason="The rendered build uses POSIX sh and make")
@pytest.mark.parametrize("workspace", ["", "packages/demo"])
def test_preparation_and_recipe_receive_the_same_workspace(tmp_path: Path, workspace: str) -> None:
    """Run actual printenv in both phases, including the repository-root sentinel."""
    recipe = tmp_path / workspace
    recipe.mkdir(parents=True, exist_ok=True)
    (recipe / "Makefile").write_text(
        "check:\n\tprintenv CORVIS_FLEET_WORKSPACE\n", encoding="utf-8"
    )
    body = LinuxDialect().build_script(
        target=tmp_path.as_posix(),
        path=workspace,
        workers=1,
        install=(InstallStep(phase="install", argv=("printenv", "CORVIS_FLEET_WORKSPACE")),),
        cache_root=(tmp_path / "cache").as_posix(),
        isolated_docker=False,
        elevated=False,
        agent="gpt6-deploy-1005",
    )
    result = subprocess.run(["/bin/sh"], input=body, text=True, capture_output=True, check=True)
    assert result.returncode == 0
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "0\n"
    lines = (tmp_path / "result.txt.log").read_text(encoding="utf-8").splitlines()
    assert lines.count(workspace or ".") == 2
