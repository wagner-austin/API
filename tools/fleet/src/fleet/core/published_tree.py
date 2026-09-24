"""The committed session-audit a hub session verb runs, never the main checkout's working tree.

WHAT WENT WRONG (MCPs board task f4cd489f). Every session verb this runner
executes (restart, revive, and both kills) invoked ``poetry -C
<mcps>/packages/session-audit run session-audit``, whose editable install
points at the MCPs MAIN CHECKOUT's working tree. Since sessions moved to one
detached worktree each (MCPs board task 60e20691) they publish with
``worktree-publish``, which pushes ``HEAD:refs/heads/main`` and moves no
local branch, so the main checkout's own ``main`` falls behind by every
push. Measured 2026-09-24 00:1xZ: the main checkout was 39 files behind
``origin/main``, and the kill-premise fix of MCPs board task cf28b0f3 had
been on ``origin/main`` for fifteen minutes while the runner still executed
the check that fix replaced. The same tree also carries other sessions'
uncommitted edits, which the verbs executed too: the hazard MCPs
``scripts/ops/run-manager-audit.ps1`` removed for the manager audit on
2026-09-07.

WHAT THIS DOES. Before a session verb, :func:`extract_published_tree` reads
the commit ``refs/remotes/origin/main`` names in the MCPs repository, writes
``git archive`` of the three trees a verb reads (session-audit's source,
mcp-shared-py's source and the source registry) to a file under this
machine's scratch root, one directory per commit, extracts it, runs
mcp-shared-py's own generator there for the one module git does not carry
(:data:`CONTRACT_HASH_SCRIPT`), and checks the extraction holds what a verb
imports. The verb then runs with the two
source trees first on ``PYTHONPATH``, which shadows the editable install
(verified that way by the manager runner, with a deliberately broken
package), and with ``--registry-dir`` at the extracted register. The
dependencies stay the checkout's session-audit environment; only the code
and the register are pinned.

WHY THE REMOTE-TRACKING REF AND NO FETCH. Every ``git push`` from any
worktree of the repository updates ``refs/remotes/origin/main`` in the one
object store they share, and ``worktree-publish`` fetches before it pushes,
so the ref is current for everything published from this machine without
the runner reaching the network. A fetch here would put GitHub and a
credential store the scheduled task's S4U logon may not be able to read
between the queue and every kill. A commit pushed from another machine
reaches the runner at the next fetch any session on this one makes; the
closing detail names the commit that ran, so which code acted is never a
guess.

NO FALLBACK TO THE WORKING TREE. When the ref does not resolve, the archive
or the extraction fails, or the extraction lacks a file a verb needs, the
job is refused by code and nothing runs: a verb run from the working tree
on the one day the extraction fails is exactly the defect this removes.

ONE DIRECTORY PER COMMIT. A repeated commit extracts over its own directory,
which rewrites the same bytes; a new commit gets a fresh directory, so a
module deleted upstream can never be imported from a previous extraction.
Directories accumulate under the system scratch root, one per commit a
session verb ran at, a few megabytes each.
"""

from __future__ import annotations

import os
import pathlib
import re
import sys
from typing import Final, TypedDict

from fleet.core import _test_hooks
from fleet.core._test_hooks import CommandResult

#: The ref a session verb's code is read from.
PUBLISHED_REF: Final = "refs/remotes/origin/main"

#: The three trees a session verb reads, as ``git archive`` pathspecs.
SESSION_AUDIT_SRC: Final = "packages/session-audit/src"
SHARED_PY_SRC: Final = "mcp-shared-py/src"
REGISTRY_DIR: Final = "mcp-shared/src/source-registry"

#: mcp-shared-py's generator for its one gitignored module. The package's
#: ``__init__`` imports ``contract_hash``, which ``make`` writes from the
#: source it fingerprints and git never carries, so an archive of the
#: source alone does not import (measured 2026-09-24: ``ModuleNotFoundError:
#: No module named 'mcp_shared_py.contract_hash'``). The script resolves the
#: package beside itself and imports only the standard library, so running
#: it from the extraction writes the module the committed source implies.
CONTRACT_HASH_SCRIPT: Final = "mcp-shared-py/scripts/generate_contract_hash.py"

ARCHIVED_PATHS: Final[tuple[str, ...]] = (
    SESSION_AUDIT_SRC,
    SHARED_PY_SRC,
    REGISTRY_DIR,
    CONTRACT_HASH_SCRIPT,
)

#: Files whose absence means the extraction cannot run a verb: session-audit's
#: entry point, the shared library's generated module (present only once the
#: generator ran), and the register a kill reads its idle window from.
REQUIRED_FILES: Final[tuple[str, ...]] = (
    f"{SESSION_AUDIT_SRC}/session_audit/cli.py",
    f"{SHARED_PY_SRC}/mcp_shared_py/contract_hash.py",
    f"{REGISTRY_DIR}/supervision.json",
)

#: The scratch-root subdirectory the extractions live in, named so a person
#: who finds it knows what left it there.
EXTRACTIONS_DIR: Final = "fleet-session-audit"

#: The tarball's name inside an extraction directory.
TARBALL_NAME: Final = "published.tar"

#: A full commit id as ``git rev-parse`` prints it.
COMMIT_PATTERN: Final = re.compile(r"^[0-9a-f]{40}$")

#: The deadline for each of the three local git and tar steps, in seconds.
#: Each reads or writes a few megabytes on this machine's own disk.
TREE_STEP_TIMEOUT_SECONDS: Final[int] = 120

#: Detail prefix when the published ref does not resolve to a commit.
REF_UNRESOLVED_CODE: Final = "SESSION_TREE_REF_UNRESOLVED"

#: Detail prefix when ``git archive`` fails.
ARCHIVE_FAILED_CODE: Final = "SESSION_TREE_ARCHIVE_FAILED"

#: Detail prefix when ``tar`` fails to extract the archive.
EXTRACT_FAILED_CODE: Final = "SESSION_TREE_EXTRACT_FAILED"

#: Detail prefix when mcp-shared-py's contract-hash generator fails.
GENERATE_FAILED_CODE: Final = "SESSION_TREE_GENERATE_FAILED"

#: Detail prefix when the extraction lacks a file a verb needs.
INCOMPLETE_CODE: Final = "SESSION_TREE_INCOMPLETE"


class PublishedTree(TypedDict):
    """The committed code one session verb runs.

    Attributes:
        commit: The full commit id the extraction was taken from, named in
            the job's closing detail.
        python_path: The ``PYTHONPATH`` value putting both extracted source
            trees ahead of the editable install.
        registry_dir: The extracted register, passed as ``--registry-dir``.
    """

    commit: str
    python_path: str
    registry_dir: str


def _step_refusal(code: str, step: str, result: CommandResult) -> str:
    """Compose the refusal for one failed step.

    Args:
        code: The detail prefix.
        step: What was run, for the reader.
        result: Its outcome.

    Returns:
        ``CODE: <step> exited <n>: <stderr>``, the stderr trimmed.
    """
    return f"{code}: {step} exited {result['returncode']}: {result['stderr'].strip()[:400]}"


def extract_published_tree(mcps_root: pathlib.Path) -> PublishedTree | str:
    """Extract the published session-audit, or say why it cannot be.

    Args:
        mcps_root: The MCPs checkout whose object store holds the ref.

    Returns:
        The extraction, or a ``CODE: message`` refusal detail for the queue
        when any step fails; nothing is run from the working tree instead.
    """
    resolved = _test_hooks.run(
        ("git", "-C", str(mcps_root), "rev-parse", "--verify", f"{PUBLISHED_REF}^{{commit}}"),
        timeout_seconds=TREE_STEP_TIMEOUT_SECONDS,
    )
    commit = resolved["stdout"].strip()
    if resolved["returncode"] != 0 or COMMIT_PATTERN.fullmatch(commit) is None:
        return (
            f"{REF_UNRESOLVED_CODE}: {PUBLISHED_REF} in {mcps_root} did not resolve to a "
            f"commit (exit {resolved['returncode']}, stdout {commit[:80]!r}, stderr "
            f"{resolved['stderr'].strip()[:200]!r}); nothing ran"
        )
    destination = _test_hooks.temp_root() / EXTRACTIONS_DIR / commit
    _test_hooks.make_directory(destination)
    tarball = destination / TARBALL_NAME
    archived = _test_hooks.run(
        (
            "git",
            "-C",
            str(mcps_root),
            "archive",
            "--format=tar",
            "-o",
            str(tarball),
            commit,
            "--",
            *ARCHIVED_PATHS,
        ),
        timeout_seconds=TREE_STEP_TIMEOUT_SECONDS,
    )
    if archived["returncode"] != 0:
        return _step_refusal(ARCHIVE_FAILED_CODE, f"git archive {commit}", archived)
    extracted = _test_hooks.run(
        ("tar", "-x", "-f", str(tarball), "-C", str(destination)),
        timeout_seconds=TREE_STEP_TIMEOUT_SECONDS,
    )
    if extracted["returncode"] != 0:
        return _step_refusal(EXTRACT_FAILED_CODE, f"tar -x {tarball}", extracted)
    generator = destination / pathlib.PurePosixPath(CONTRACT_HASH_SCRIPT)
    generated = _test_hooks.run(
        (sys.executable, str(generator)),
        timeout_seconds=TREE_STEP_TIMEOUT_SECONDS,
    )
    if generated["returncode"] != 0:
        return _step_refusal(GENERATE_FAILED_CODE, str(generator), generated)
    missing = [
        required
        for required in REQUIRED_FILES
        if not _test_hooks.file_exists(destination / pathlib.PurePosixPath(required))
    ]
    if missing:
        return (
            f"{INCOMPLETE_CODE}: the extraction of {commit} at {destination} lacks "
            f"{', '.join(missing)}; nothing ran"
        )
    return PublishedTree(
        commit=commit,
        python_path=os.pathsep.join(
            (
                str(destination / pathlib.PurePosixPath(SESSION_AUDIT_SRC)),
                str(destination / pathlib.PurePosixPath(SHARED_PY_SRC)),
            )
        ),
        registry_dir=str(destination / pathlib.PurePosixPath(REGISTRY_DIR)),
    )


__all__ = [
    "ARCHIVED_PATHS",
    "ARCHIVE_FAILED_CODE",
    "COMMIT_PATTERN",
    "CONTRACT_HASH_SCRIPT",
    "EXTRACTIONS_DIR",
    "EXTRACT_FAILED_CODE",
    "GENERATE_FAILED_CODE",
    "INCOMPLETE_CODE",
    "PUBLISHED_REF",
    "REF_UNRESOLVED_CODE",
    "REGISTRY_DIR",
    "REQUIRED_FILES",
    "SESSION_AUDIT_SRC",
    "SHARED_PY_SRC",
    "TARBALL_NAME",
    "TREE_STEP_TIMEOUT_SECONDS",
    "PublishedTree",
    "extract_published_tree",
]
