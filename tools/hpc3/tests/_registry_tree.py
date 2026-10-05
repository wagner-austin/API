"""A copy of the committed registry and research index for a test to register into.

``hpc3-register`` WRITES a workspace document and the index, so a test that
drove it against the real tree would race every test reading them. Copying the
real documents rather than inventing small ones keeps the point of the test:
the command lands a project beside the eight already registered, through the
same index the next session reads, and the table it writes is checked against
what that whole registry declares.
"""

from __future__ import annotations

import pathlib
import shutil

from hpc3.core.index_sections import REGISTERED_HEADING

REAL_ROOT = pathlib.Path(__file__).resolve().parents[3]
"""The monorepo checkout this suite runs in."""

NEWCOMER = "newcomer"
"""The project every registration test registers."""

NEWCOMER_IMAGE = "/pub/wagnera3/newcomer/images/v1/newcomer.sif"
"""Where its built image sits on the cluster."""

NEWCOMER_DIGEST = "3" * 64
"""What ``sha256sum`` reports for that image."""


def copy_tree(destination: pathlib.Path) -> pathlib.Path:
    """Copy every workspace document and the research index under a new root.

    Args:
        destination: Directory to become the copy's monorepo root.

    Returns:
        That root, holding ``tools/hpc3/runs/hpc3*.json``, ``docs/RESEARCH.md``
        and an empty ``clients/Newcomer`` for the project's repo.
    """
    runs = runs_of(destination)
    runs.mkdir(parents=True)
    for path in sorted((REAL_ROOT / "tools" / "hpc3" / "runs").glob("hpc3*.json")):
        shutil.copyfile(path, runs / path.name)
    index_of(destination).parent.mkdir()
    shutil.copyfile(REAL_ROOT / "docs" / "RESEARCH.md", index_of(destination))
    repo_of(destination).mkdir(parents=True)
    return destination


def runs_of(root: pathlib.Path) -> pathlib.Path:
    """Name the copy's workspace directory.

    Args:
        root: The copy's monorepo root.

    Returns:
        ``tools/hpc3/runs`` under it.
    """
    return root / "tools" / "hpc3" / "runs"


def index_of(root: pathlib.Path) -> pathlib.Path:
    """Name the copy's research index.

    Args:
        root: The copy's monorepo root.

    Returns:
        ``docs/RESEARCH.md`` under it.
    """
    return root / "docs" / "RESEARCH.md"


def repo_of(root: pathlib.Path) -> pathlib.Path:
    """Name the copy's repository for the newcomer.

    Args:
        root: The copy's monorepo root.

    Returns:
        ``clients/Newcomer`` under it.
    """
    return root / "clients" / "Newcomer"


def write_section(root: pathlib.Path, project: str) -> None:
    """Write a project's section into the copy's registered part, by hand.

    The step registration refuses without, done the way a person does it:
    a heading and prose, placed directly under the registered part's heading.

    Args:
        root: The copy's monorepo root.
        project: The project the section describes.
    """
    index = index_of(root)
    text = index.read_text(encoding="utf-8")
    section = f"{REGISTERED_HEADING}\n\n### `{project}` — what it measures\n\nProse.\n"
    index.write_text(text.replace(f"{REGISTERED_HEADING}\n", section, 1), encoding="utf-8")


__all__ = [
    "NEWCOMER",
    "NEWCOMER_DIGEST",
    "NEWCOMER_IMAGE",
    "REAL_ROOT",
    "copy_tree",
    "index_of",
    "repo_of",
    "runs_of",
    "write_section",
]
