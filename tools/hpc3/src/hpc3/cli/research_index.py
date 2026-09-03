"""CLI: render the research index's generated project table.

Usage:
    hpc3-research-index --check    # exit 1 when the committed block is stale
    hpc3-research-index --write    # rewrite the block in docs/RESEARCH.md

NEITHER IS THE DEFAULT, and a bare invocation refuses, which is the
convention every command in this package follows and which
``test_cli_entrypoint_shape`` enforces. Defaulting to the checking form would
be harmless; defaulting at all would mean a caller who typed the wrong thing
got an action they did not name. Writing is never the default for a stronger
reason: a command that mutates a tracked document by default is one somebody
runs to make a test pass without reading what changed, and this block exists
because a number nobody reread was wrong for a day.
"""

from __future__ import annotations

import pathlib
import sys
from collections.abc import Sequence

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from hpc3.cli import _fatal
from hpc3.contracts.workspace import ProjectConfig, decode_workspace
from hpc3.core.research_index import (
    REGENERATE_HINT,
    extract_projects_block,
    render_projects_block,
    replace_projects_block,
)

WRITE_FLAG = "--write"
CHECK_FLAG = "--check"

#: The flags, and the ONE place the set is written.
FLAGS: tuple[str, ...] = (CHECK_FLAG, WRITE_FLAG)


def runs_directory() -> pathlib.Path:
    """Locate the workspace documents.

    Returns:
        The ``runs`` directory of this package.
    """
    return pathlib.Path(__file__).resolve().parents[3] / "runs"


def index_path() -> pathlib.Path:
    """Locate the research index.

    Returns:
        ``docs/RESEARCH.md`` at the monorepo root. It names work in other
        repositories, so it lives above the tool that submits some of it.
    """
    return pathlib.Path(__file__).resolve().parents[5] / "docs" / "RESEARCH.md"


def declared_projects(runs: pathlib.Path) -> dict[str, ProjectConfig]:
    """Read every project the committed workspaces declare.

    Args:
        runs: Directory holding the workspace documents.

    Returns:
        Every declared project, keyed by name.

    Raises:
        ValueError: If two workspaces declare the same project, which leaves
            no answer to which one governs a run naming it.
    """
    projects: dict[str, ProjectConfig] = {}
    for path in sorted(runs.glob("*.json")):
        document = narrow_json_to_dict(load_json_str(path.read_text(encoding="utf-8")))
        if "projects" not in document:
            continue
        workspace = decode_workspace(document, config_dir=runs)
        for name, config in workspace["projects"].items():
            if name in projects:
                raise ValueError(f"project {name!r} is declared twice, one place is {path.name}")
            projects[name] = config
    return projects


def main(argv: Sequence[str] | None = None) -> int:
    """Render the block, and either write it or check it.

    Args:
        argv: Arguments excluding the program name. Defaults to the process
            arguments.

    Returns:
        Exit code 0 when the file already matches or was written, 1 when it
        is stale.

    Raises:
        ValueError: If an unknown argument is given, if neither flag is
            given, or if both are. A bare invocation naming no action is a
            caller who has not said what they want, and guessing for them is
            how a document gets rewritten by somebody who meant to check it.
    """
    tokens = list(argv) if argv is not None else list(sys.argv[1:])
    unknown = [token for token in tokens if token not in FLAGS]
    if unknown:
        raise ValueError(f"unknown argument(s) {unknown}; known flags: {FLAGS}")
    if len(set(tokens)) != 1:
        raise ValueError(f"name exactly one of {FLAGS}")

    block = render_projects_block(declared_projects(runs_directory()))
    path = index_path()
    text = path.read_text(encoding="utf-8")

    if WRITE_FLAG in tokens:
        path.write_text(replace_projects_block(text, block), encoding="utf-8")
        sys.stdout.write(f"wrote the project table into {path}\n")
        return 0

    if extract_projects_block(text) == block:
        sys.stdout.write("the project table matches the registry\n")
        return 0

    sys.stdout.write(
        f"the project table in {path} is stale; run `{REGENERATE_HINT}`\n\n{block}\n"
    )
    return 1


def entrypoint() -> None:
    """Console-script entry point.

    Refusals travel through ``_fatal.run`` like every other command here, so
    a ValueError becomes EXIT_REFUSED with its message on stderr rather than
    a traceback.

    Raises:
        SystemExit: Always, carrying :func:`main`'s exit code.
    """
    raise SystemExit(_fatal.run(main))


__all__ = [
    "CHECK_FLAG",
    "FLAGS",
    "WRITE_FLAG",
    "declared_projects",
    "entrypoint",
    "index_path",
    "main",
    "runs_directory",
]


if __name__ == "__main__":
    entrypoint()
