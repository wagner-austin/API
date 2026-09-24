"""The published-tree extraction a session verb makes first, scripted for the suites.

Shared by ``test_published_tree`` (the extraction itself) and
``test_agent_restart`` (the tick that runs it before every session verb), so
the four commands, their order and the files a successful extraction leaves
are written down once. The extraction's files are planted for real under a
per-test scratch root: the existence check reads the disk through the real
``file_exists`` hook, so what a test plants is what the code finds.
"""

from __future__ import annotations

import os
import pathlib
import sys
from typing import Final

from fleet.core import _test_hooks, published_tree
from fleet.core.published_tree import PublishedTree
from tests.conftest import FakeTempRoot, ok

#: The commit the scripted ``git rev-parse`` answers with: the commit the
#: live extraction resolved on 2026-09-24, so the shape is a real one.
COMMIT: Final = "6e920e3e17f7ed9275422b703c33ca4325c3f45b"


def extraction_dir(scratch: pathlib.Path) -> pathlib.Path:
    """Where the extraction of :data:`COMMIT` lands under a scratch root.

    Args:
        scratch: The pinned scratch root.

    Returns:
        The per-commit directory.
    """
    return scratch / published_tree.EXTRACTIONS_DIR / COMMIT


def pin_scratch(scratch: pathlib.Path) -> None:
    """Pin the scratch root the extraction writes under to a test directory.

    Args:
        scratch: The directory to hand out.
    """
    _test_hooks.temp_root = FakeTempRoot(scratch)


def plant_extraction(scratch: pathlib.Path) -> PublishedTree:
    """Pin the scratch root and leave the files a successful extraction leaves.

    Args:
        scratch: The per-test scratch root.

    Returns:
        The tree :func:`~fleet.core.published_tree.extract_published_tree`
        returns once the four scripted commands succeed.
    """
    pin_scratch(scratch)
    destination = extraction_dir(scratch)
    for required in published_tree.REQUIRED_FILES:
        path = destination / pathlib.PurePosixPath(required)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("", encoding="utf-8")
    return expected_tree(scratch)


def expected_tree(scratch: pathlib.Path) -> PublishedTree:
    """The tree an extraction of :data:`COMMIT` under this root describes.

    Args:
        scratch: The pinned scratch root.

    Returns:
        The commit, the two source trees joined for ``PYTHONPATH``, and the
        extracted register.
    """
    destination = extraction_dir(scratch)
    return PublishedTree(
        commit=COMMIT,
        python_path=(
            f"{destination / 'packages' / 'session-audit' / 'src'}{os.pathsep}"
            f"{destination / 'mcp-shared-py' / 'src'}"
        ),
        registry_dir=str(destination / "mcp-shared" / "src" / "source-registry"),
    )


def extraction_calls(mcps: pathlib.Path, scratch: pathlib.Path) -> list[tuple[str, ...]]:
    """The four commands an extraction runs, in order.

    Args:
        mcps: The MCPs checkout.
        scratch: The pinned scratch root.

    Returns:
        ``git rev-parse``, ``git archive``, ``tar -x`` and the contract-hash
        generator, as argv tuples.
    """
    destination = extraction_dir(scratch)
    tarball = destination / published_tree.TARBALL_NAME
    return [
        ("git", "-C", str(mcps), "rev-parse", "--verify", "refs/remotes/origin/main^{commit}"),
        (
            "git",
            "-C",
            str(mcps),
            "archive",
            "--format=tar",
            "-o",
            str(tarball),
            COMMIT,
            "--",
            "packages/session-audit/src",
            "mcp-shared-py/src",
            "mcp-shared/src/source-registry",
            "mcp-shared-py/scripts/generate_contract_hash.py",
        ),
        ("tar", "-x", "-f", str(tarball), "-C", str(destination)),
        (
            sys.executable,
            str(destination / "mcp-shared-py" / "scripts" / "generate_contract_hash.py"),
        ),
    ]


def extraction_replies() -> list[_test_hooks.CommandResult]:
    """The four successful answers, in the order :func:`extraction_calls` asks.

    Returns:
        The commit from ``rev-parse``, silence from ``archive`` and ``tar``,
        and the generator's own summary line.
    """
    return [
        ok(f"{COMMIT}\n"),
        ok(""),
        ok(""),
        ok("Wrote contract_hash.py (hash=0123456789ab..., files=212)\n"),
    ]


__all__ = [
    "COMMIT",
    "expected_tree",
    "extraction_calls",
    "extraction_dir",
    "extraction_replies",
    "pin_scratch",
    "plant_extraction",
]
