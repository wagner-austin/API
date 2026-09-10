"""Reading the run documents as GIT holds them, shared by the modules that audit them.

``test_committed_runs`` asks whether each committed record still decodes;
``test_committed_run_images`` asks whether the image each one names actually
carries the plan it invokes. They were one module until it passed the
600-line ceiling, and the reader below is the piece they must not fork,
because the whole value of both is that they read the same set.

WHY THIS READS HEAD AND NOT THE WORKING DIRECTORY. ``.gitignore`` ignores
``tools/hpc3/runs/*`` and then RE-INCLUDES several families by pattern, which
is why a tracked set exists at all and why the tracked and on-disk counts can
diverge far without either looking wrong. Globbing the filesystem measured a
set that exists only on the machine that wrote it: on 2026-09-07 a working
tree held 488 JSON documents against 259 at HEAD, and a re-measure the same
day read 492 and 263 as four more landed.

That is not hypothetical. A floor calibrated at 138 locally arrived on CI as
``assert 36 >= 100`` (run 34104178998). Every developer's ``make check`` was
green, because every working tree is self-consistent and CI is the only
reader that starts from a clean checkout.

THE COST OF READING HEAD, STATED SO NEITHER CALLER FORGETS IT. These audits
convict a document on the check AFTER it lands, not at the commit that lands
it. That is a deliberate trade -- a worktree read is the one that was
measured wrong -- and it is still in time for what matters, since a run
document is committed well before its GPU hours are spent.
"""

from __future__ import annotations

import io
import pathlib
import subprocess
import tarfile
import tempfile

from platform_core.json_utils import JSONValue, load_json_str

REPO = pathlib.Path(__file__).parents[3]
"""The monorepo root, which is where ``git`` must be invoked from."""

RUNS_IN_REPO = "tools/hpc3/runs"
"""``runs/`` as git spells it, which is the only spelling git accepts."""


def documents() -> list[tuple[str, dict[str, JSONValue]]]:
    """Read every JSON object COMMITTED under ``runs/``.

    ``git archive HEAD`` rather than a per-file ``git show``: one subprocess
    instead of hundreds, and it works on the shallow clone
    ``actions/checkout`` produces by default, because HEAD's tree is present
    even at depth 1. An older revision would not be, which is a separate trap
    this repo has already paid for once.

    Returns:
        Each document's filename and parsed body, in filename order.
    """
    archive = subprocess.run(
        ["git", "archive", "HEAD", "--", RUNS_IN_REPO],
        cwd=REPO,
        capture_output=True,
        check=True,
    ).stdout
    found: list[tuple[str, dict[str, JSONValue]]] = []
    with tempfile.TemporaryDirectory() as scratch:
        root = pathlib.Path(scratch)
        with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
            tar.extractall(root, filter="data")
        for path in sorted((root / RUNS_IN_REPO).glob("*.json")):
            document = load_json_str(path.read_text(encoding="utf-8"))
            if isinstance(document, dict):
                found.append((path.name, document))
    return found


def submissions() -> list[tuple[str, str, dict[str, JSONValue]]]:
    """Select the documents that are submissions rather than configuration.

    A submission is identified by naming a ``project``, which is what
    :func:`~hpc3.contracts.run.resolve_run` requires first. That predicate
    excludes the workspaces themselves, image specifications, and any
    document predating the field -- without an exemption list, which is the
    point: a filename is not a reason.

    Returns:
        Each submission's filename, project name, and body.

    Raises:
        TypeError: If a document's ``project`` is not a string.
    """
    found: list[tuple[str, str, dict[str, JSONValue]]] = []
    for name, document in documents():
        project = document.get("project")
        if project is None:
            continue
        if not isinstance(project, str):
            raise TypeError(f"{name}: 'project' must be a string")
        found.append((name, project, document))
    return found


__all__ = ["REPO", "RUNS_IN_REPO", "documents", "submissions"]
