"""The images tools/maketools-execution builds before its timed check (MCPs board task eb6586e4).

The fleet runs that project's ``images`` target as an install step, so the
deploy cases' own builds read a warm build cache and the check's budget
measures the deploys: on loki on 2026-11-01 with an empty build cache the
eight cases took 460 to 654 s. A build the target misses is a build the
check pays for again, so these cases hold the target to what the cases
deploy: one target per service in :data:`tests.test_service_entry.SERVICES`,
each running exactly the builds that service's compose file declares.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Final

from tests.test_service_entry import REPO_ROOT, SERVICES

#: The project's recipe.
MAKEFILE: Final[Path] = REPO_ROOT / "tools" / "maketools-execution" / "Makefile"

#: One build block of a service's compose file: the repository root as its
#: context, its Dockerfile, and the stage it targets when it names one.
BUILD: Final[re.Pattern[str]] = re.compile(
    r"^ +build:\n +context: \.\./\.\.\n +dockerfile: (\S+)\n(?: +target: (\S+)\n)?", re.MULTILINE
)


def _declared_builds(service: str) -> list[str]:
    """The ``docker build`` line for each build a service's compose file declares.

    Args:
        service: The service directory's name.

    Returns:
        One line per build block, as the target runs it from its directory.
    """
    text = (REPO_ROOT / "services" / service / "docker-compose.yml").read_text(encoding="utf-8")
    blocks = [f"{found.group(0)}" for found in BUILD.finditer(text)]
    assert len(blocks) == text.count("build:"), f"{service}: a build block of another shape"
    lines: list[str] = []
    for block in blocks:
        parts = [line.strip().partition(": ") for line in block.splitlines()[1:]]
        fields = {key: value for key, _, value in parts}
        target = f" --target {fields['target']}" if "target" in fields else ""
        lines.append(f"docker build --quiet --file ../../{fields['dockerfile']}{target} ../..")
    return lines


def _recipe(target: str) -> list[str]:
    """The recipe lines of one Makefile target.

    Args:
        target: The target's name.

    Returns:
        Its tab-indented lines, without the tab, up to the next blank line.
    """
    lines = MAKEFILE.read_text(encoding="utf-8").splitlines()
    start = lines.index(f"{target}:") + 1
    recipe: list[str] = []
    for line in lines[start:]:
        if not line.startswith("\t"):
            break
        recipe.append(line.removeprefix("\t"))
    return recipe


def test_the_images_target_names_one_target_per_deployed_service() -> None:
    text = MAKEFILE.read_text(encoding="utf-8")
    listed = text.split("\nIMAGES := ", 1)[1].split("\n\n", 1)[0]

    assert listed.replace("\\\n", " ").split() == sorted(f"image-{service}" for service in SERVICES)
    assert "images: $(IMAGES)" in text.splitlines()


def test_each_service_target_runs_the_builds_its_compose_file_declares() -> None:
    for service in sorted(SERVICES):
        assert _recipe(f"image-{service}") == _declared_builds(service), service


def test_the_workers_are_built_beside_their_api() -> None:
    """Five of the eight run a worker from a second stage of the same file."""
    workers = sorted(service for service, entry in SERVICES.items() if entry["worker"])

    assert workers == sorted(service for service in SERVICES if len(_declared_builds(service)) == 2)
    assert [_recipe(f"image-{service}")[-1].split()[-2] for service in workers] == ["worker"] * 5
