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
from typing import Final

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from hpc3.cli import _fatal
from hpc3.contracts.project import ProjectConfig
from hpc3.contracts.workspace import decode_workspace
from hpc3.core import _test_hooks as core_hooks
from hpc3.core.research_index import (
    REGENERATE_HINT,
    extract_projects_block,
    image_digest_claims,
    ledger_state_claims,
    parse_review_markers,
    render_projects_block,
    replace_projects_block,
    stale_review_claims,
)

WRITE_FLAG = "--write"
CHECK_FLAG = "--check"

#: The flags, and the ONE place the set is written.
FLAGS: tuple[str, ...] = (CHECK_FLAG, WRITE_FLAG)

#: What a reader is told when the index restates a value it does not own. One
#: message for both classes because the remedy is one remedy: the ledger is
#: machine-local so a count off it is uncheckable, the image digests are
#: rendered so a copy of one is redundant, and in all three live instances
#: UPDATING the restatement is exactly what had been done before and what left
#: it stale again. The per-claim lines above it name which value and which
#: project; this says what to do about it.
CLAIM_GUIDANCE: Final[str] = (
    "each line above restates a value this file does not own; delete the restatement "
    "rather than updating it, because updating is what left the last three stale\n"
)


def _read_document(path: pathlib.Path) -> str:
    """Read a text document with its line endings normalised.

    ``read_bytes`` is the package's file seam and does no newline
    translation, unlike ``pathlib.read_text``. Git checks this repository out
    with CRLF on Windows while the renderer emits LF, so a byte comparison
    reported a document as stale whose every line was already correct --
    a false verdict that would have had someone rewrite a correct file.

    Args:
        path: The document.

    Returns:
        Its text, with CRLF collapsed to LF.
    """
    return core_hooks.read_bytes(path).decode("utf-8").replace("\r\n", "\n")


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
        document = narrow_json_to_dict(load_json_str(_read_document(path)))
        if "projects" not in document:
            continue
        workspace = decode_workspace(document, config_dir=runs)
        for name, config in workspace["projects"].items():
            if name in projects:
                raise ValueError(f"project {name!r} is declared twice, one place is {path.name}")
            projects[name] = config
    return projects


def tracked_counts(globs: tuple[str, ...]) -> dict[str, int]:
    """Count the tracked files each glob matches.

    Asked of git rather than of the filesystem, and the distinction is the
    point. A count over what happens to be on this disk would be the same
    uncheckable claim :func:`~hpc3.core.research_index.ledger_state_claims`
    refuses; a count over what is COMMITTED is one any clone reproduces, which
    is what makes the marker worth failing a build over.

    Args:
        globs: Pathspecs to count, as written in the markers.

    Returns:
        The number of tracked files matching each glob.

    Raises:
        RuntimeError: If git refuses a pathspec. A glob nobody can resolve
            would otherwise count zero and read as "the evidence was deleted",
            which is a different and much louder claim than "this is a typo".
    """
    root = index_path().parents[1]
    counts: dict[str, int] = {}
    for glob in globs:
        result = core_hooks.run(["git", "-C", str(root), "ls-files", "--", glob])
        if result["returncode"] != 0:
            raise RuntimeError(
                f"git refused the review marker pathspec {glob!r}: {result['stderr']}"
            )
        counts[glob] = len([line for line in result["stdout"].splitlines() if line])
    return counts


def main(argv: Sequence[str] | None = None) -> int:
    """Render the block, and either write it or check it.

    Args:
        argv: Arguments excluding the program name. Defaults to the process
            arguments.

    Returns:
        Exit code 0 when the file already matches or was written and asserts
        no machine-local run state, 1 when it is stale or does assert some.

        ``--write`` returns 1 on an asserted claim even though it wrote the
        block successfully, because the claim is PROSE and writing cannot fix
        it. Reporting 0 there would let a caller who runs the writing form
        conclude the document is clean when the half no generator owns is
        not.

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

    projects = declared_projects(runs_directory())
    block = render_projects_block(projects)
    path = index_path()
    text = _read_document(path)

    markers = parse_review_markers(text)
    claims = (
        ledger_state_claims(text)
        + image_digest_claims(text, projects)
        + stale_review_claims(markers, tracked_counts(tuple(glob for glob, _ in markers)))
    )
    for claim in claims:
        sys.stdout.write(f"{path}: {claim}\n")
    if claims:
        sys.stdout.write(CLAIM_GUIDANCE)

    if WRITE_FLAG in tokens:
        core_hooks.write_text(path, replace_projects_block(text, block))
        sys.stdout.write(f"wrote the project table into {path}\n")
        return 1 if claims else 0

    if extract_projects_block(text) == block:
        if claims:
            return 1
        sys.stdout.write("the project table matches the registry\n")
        return 0

    sys.stdout.write(f"the project table in {path} is stale; run `{REGENERATE_HINT}`\n\n{block}\n")
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
    "CLAIM_GUIDANCE",
    "FLAGS",
    "WRITE_FLAG",
    "declared_projects",
    "entrypoint",
    "index_path",
    "main",
    "runs_directory",
    "tracked_counts",
]


if __name__ == "__main__":
    entrypoint()
