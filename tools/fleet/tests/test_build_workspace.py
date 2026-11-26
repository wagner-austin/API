"""Execute a rendered build and verify preparation receives its project directory."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from fleet.contracts.source import InstallStep
from fleet.core import dialect


@pytest.mark.parametrize("workspace", ["", "packages/demo"])
@pytest.mark.parametrize("workers", [1, 3, 6])
def test_preparation_and_recipe_receive_the_same_workspace(
    tmp_path: Path, workspace: str, workers: int
) -> None:
    """Execute both phases with the project and granted Python/TypeScript worker budget."""
    recipe = tmp_path / workspace
    recipe.mkdir(parents=True, exist_ok=True)
    (recipe / "Makefile").write_text(
        "check:\n\tprintenv CORVIS_FLEET_WORKSPACE PYTEST_XDIST_AUTO_NUM_WORKERS "
        "CORVIS_TEST_MAX_WORKERS\n",
        encoding="utf-8",
    )
    body = dialect.build_script(
        target=tmp_path.as_posix(),
        path=workspace,
        workers=workers,
        ram_gb=0.5 * workers,
        install=(
            InstallStep(
                phase="install",
                argv=(
                    "printenv",
                    "CORVIS_FLEET_WORKSPACE",
                    "PYTEST_XDIST_AUTO_NUM_WORKERS",
                    "CORVIS_TEST_MAX_WORKERS",
                ),
            ),
        ),
        cache_root=(tmp_path / "cache").as_posix(),
        isolated_docker=False,
        agent="gpt6-deploy-1005",
        loopback_unit=None,
        artifact=None,
    )
    result = subprocess.run(["/bin/sh"], input=body, text=True, capture_output=True, check=True)
    assert result.returncode == 0
    assert (tmp_path / "result.txt").read_text(encoding="utf-8") == "0\n"
    lines = (tmp_path / "result.txt.log").read_text(encoding="utf-8").splitlines()
    assert lines.count(workspace or ".") == 2
    assert lines.count(str(workers)) == 4
    assert "FLEET_ARTIFACT: none" in lines


@pytest.mark.parametrize(
    ("booked", "printed"),
    [(1.85, "1.85"), (7 * 0.35, "2.4499999999999997"), (3.0, "3.0")],
)
def test_the_recipe_reads_the_booked_memory_as_the_double_admission_weighed(
    tmp_path: Path, booked: float, printed: str
) -> None:
    """MCPs board task 330b0465: the mutation step sizes StrykerJS to this figure.

    Each printed value is Python's shortest round-tripping spelling of the
    double, worked by hand: 7 x 0.35 is not 2.45 in binary.
    """
    (tmp_path / "Makefile").write_text("check:\n\tprintenv CORVIS_TEST_RAM_GB\n", encoding="utf-8")
    body = dialect.build_script(
        target=tmp_path.as_posix(),
        path="",
        workers=3,
        ram_gb=booked,
        install=(),
        cache_root=(tmp_path / "cache").as_posix(),
        isolated_docker=False,
        agent="opus-mcps-1125-8f88",
        loopback_unit=None,
        artifact=None,
    )
    subprocess.run(["/bin/sh"], input=body, text=True, capture_output=True, check=True)
    lines = (tmp_path / "result.txt.log").read_text(encoding="utf-8").splitlines()
    assert printed in lines
    assert float(printed) == booked
