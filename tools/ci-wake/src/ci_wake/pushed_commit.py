"""The commit a push lands and the workflows in it, asked of git.

THE SHA GIT HANDS ``pre-push`` IS AN OBJECT, NOT ALWAYS A COMMIT (MCPs board
task 24547f50). For an annotated tag it is the tag object: on 2026-10-11
the push of ``archive/dx-mod`` enrolled 611e26a3, while the commit the tag
names, and the ``head_sha`` any run for it would carry, is 4b4883ad. A row
under the tag object's sha can match no run, so the enrolment peels the
pushed object to its commit before recording it.

GIT RUNS WHERE THE HOOK DOES. Git starts ``pre-push`` in the root of the
working tree being pushed, and the enrolment inherits that directory, so
these commands ask the repository being pushed and no other. They read
objects only: nothing here writes to the repository.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from platform_core.error_codes_tooling import CiWakeErrorCode
from platform_core.errors import AppError

from ci_wake import _test_hooks
from ci_wake.enrolment import require_sha
from ci_wake.workflow_triggers import Workflow, read_workflow

#: Seconds any one git read may take. Each reads a handful of objects from
#: the local repository; one this slow is a repository in trouble, and the
#: push it would block is better refused with git's words than left hanging.
GIT_TIMEOUT_SECONDS: Final = 30

#: Where GitHub reads workflows from: files directly in this directory.
WORKFLOWS_DIR: Final = ".github/workflows"

#: The file extensions GitHub reads as workflows there.
WORKFLOW_SUFFIXES: Final[tuple[str, ...]] = (".yml", ".yaml")


def _git(args: Sequence[str], purpose: str) -> str:
    """Run one read-only git command in the repository being pushed.

    Args:
        args: Its arguments after ``git``.
        purpose: What the read is for, for the refusal.

    Returns:
        Its standard output.

    Raises:
        AppError: ``PUSHED_COMMIT_UNREADABLE`` when git exits non-zero,
            carrying its own stderr.
    """
    completed = _test_hooks.run_process(["git", *args], timeout=GIT_TIMEOUT_SECONDS)
    if completed.returncode != 0:
        raise AppError(
            code=CiWakeErrorCode.PUSHED_COMMIT_UNREADABLE,
            message=(
                f"git {' '.join(args)} exited {completed.returncode} while {purpose}: "
                f"{completed.stderr.strip()}"
            ),
        )
    return completed.stdout


def peel_commit(pushed: str) -> str:
    """The commit a pushed object names: itself, or the commit a tag points to.

    Args:
        pushed: The full sha git handed the hook.

    Returns:
        The commit's full sha.

    Raises:
        AppError: ``PUSHED_COMMIT_UNREADABLE`` when it names no commit;
            ``ENROLMENT_FIELD_MALFORMED`` when git answers with something
            that is not a full sha.
    """
    purpose = f"peeling the pushed object {pushed} to a commit"
    return require_sha(_git(["rev-parse", "--verify", f"{pushed}^{{commit}}"], purpose).strip())


def read_workflows(commit: str) -> tuple[Workflow, ...]:
    """Every workflow GitHub would read for a push of this commit.

    Args:
        commit: The commit's full sha.

    Returns:
        Each ``.yml`` or ``.yaml`` file directly in ``.github/workflows``,
        read, in git's order; empty when the directory does not exist.

    Raises:
        AppError: ``PUSHED_COMMIT_UNREADABLE`` when git cannot list or read
            one; ``WORKFLOW_TRIGGER_UNREAD`` from the reader.
    """
    listing = _git(
        ["ls-tree", "-z", commit, "--", f"{WORKFLOWS_DIR}/"],
        f"listing {WORKFLOWS_DIR} in {commit}",
    )
    workflows: list[Workflow] = []
    for entry in listing.split("\0"):
        if entry == "":
            continue
        # ``<mode> SP <type> SP <object> TAB <path>``, as ls-tree documents.
        header, _, path = entry.partition("\t")
        _, kind, blob = header.split(" ")
        name = path.rsplit("/", 1)[-1]
        if kind != "blob" or not name.endswith(WORKFLOW_SUFFIXES):
            continue
        text = _git(["cat-file", "blob", blob], f"reading {path} in {commit}")
        workflows.append(read_workflow(name, text))
    return tuple(workflows)


__all__ = ["GIT_TIMEOUT_SECONDS", "WORKFLOWS_DIR", "peel_commit", "read_workflows"]
