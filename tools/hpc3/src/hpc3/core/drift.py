"""Comparing the code an image will run with the code its submitter has.

:mod:`hpc3.core.entrypoint` refuses a module the image LACKS. A module it
HAS is admitted, because reproducing an old command under its old image must
stay legal, and that leaves the case that produces numbers rather than a
crash: the module's source has changed in the submitter's repository since
the commit the image was built from, so the run executes older code than the
person reading its numbers is likely to assume (board task 7aab32ef). This
measures that, with the submitter's own git, and says what it found; it
refuses nothing about drift itself.

WHAT IS COMPARED. For each ``python -m`` module that resolved inside the
image's environment, the source of its top-level package as the repository
tracks it at HEAD: every tracked ``<top>/__init__.py`` directory, or a
``<top>.py`` file, wherever it sits. Both the image's commit and HEAD are
read from the project's ``repo``, and ``git diff`` between them names what
changed under those paths. HEAD, not the working tree: a commit is a fixed
thing a ledger row can name, and an uncommitted edit is not.

WHAT IS NOT, AND SAYS SO. A module that resolved OUTSIDE the environment
runs the command's own checkout, which the image's commit does not describe
(:mod:`hpc3.core.code_claim` holds that checkout to the commit a run
declares). A package the repository tracks no source for is a third-party
one. A run with no image has no build commit at all. Each is recorded with
its reason, so an empty change list never reads as a comparison that was not
made. The packages a module imports are not compared; its own package is.

WHAT IS REFUSED: an image commit the repository does not hold
(``ENTRYPOINT_IMAGE_COMMIT_UNKNOWN``), and git failing to answer once it
does (``ENTRYPOINT_DRIFT_UNREADABLE``). Admitting either would record a
comparison nobody made.
"""

from __future__ import annotations

import pathlib
from collections.abc import Sequence

from platform_core.errors import AppError, Hpc3ErrorCode

from hpc3.contracts.drift import ImageDrift, ModuleDrift
from hpc3.core import _test_hooks
from hpc3.core.entrypoint import EntrypointReading, ModuleReading, invocations

#: How many changed paths a warning names before it counts the rest.
WARNING_PATHS = 5


def _git(repo: pathlib.Path, *args: str) -> str:
    """Run one git command in the project's repository.

    Args:
        repo: The repository.
        *args: Git's arguments.

    Returns:
        Its standard output.

    Raises:
        AppError: With ``ENTRYPOINT_DRIFT_UNREADABLE`` carrying git's own
            complaint when it exits non-zero.
    """
    result = _test_hooks.run(["git", "-C", str(repo), *args])
    if result["returncode"] != 0:
        raise AppError(
            Hpc3ErrorCode.ENTRYPOINT_DRIFT_UNREADABLE,
            f"git {' '.join(args)} in {repo} exited {result['returncode']}: "
            f"{result['stderr'].strip()}. Without its answer this run's code cannot be "
            "compared with the submitter's.",
        )
    return result["stdout"]


def source_paths(tracked: Sequence[str], top: str) -> list[str]:
    """Find where a repository keeps a top-level package's source.

    Args:
        tracked: Every path the repository tracks at HEAD.
        top: The top-level package or module name.

    Returns:
        Each directory holding a tracked ``<top>/__init__.py`` and each
        tracked ``<top>.py``, sorted and without repeats. Empty when the
        repository has none, which makes it a third-party package.
    """
    found: set[str] = set()
    for path in tracked:
        parts = path.split("/")
        if parts[-1] == "__init__.py" and len(parts) > 1 and parts[-2] == top:
            found.add("/".join(parts[:-1]))
        if parts[-1] == f"{top}.py":
            found.add(path)
    return sorted(found)


def _module_drift(
    module: ModuleReading,
    *,
    repo: pathlib.Path,
    commit: str,
    head: str,
    tracked: Sequence[str],
) -> ModuleDrift:
    """Compare one module's source between the image's commit and HEAD.

    Args:
        module: Where the module resolved in the image.
        repo: The project's repository.
        commit: The image's commit.
        head: The repository's HEAD.
        tracked: Every path the repository tracks at HEAD.

    Returns:
        What was compared and what changed, or why nothing was.

    Raises:
        AppError: With ``ENTRYPOINT_DRIFT_UNREADABLE`` if git cannot diff.
    """
    top = module["module"].split(".")[0]
    if not module["in_environment"]:
        reason = (
            f"resolved from {module['location']}, outside the image's environment, so the "
            "image's commit does not describe its code"
        )
        return ModuleDrift(
            directory=module["directory"],
            module=module["module"],
            compared=[],
            changed=[],
            not_compared=reason,
        )
    compared = source_paths(tracked, top)
    if compared == []:
        return ModuleDrift(
            directory=module["directory"],
            module=module["module"],
            compared=[],
            changed=[],
            not_compared=f"{repo} tracks no source for {top!r} at HEAD",
        )
    changed = _git(repo, "diff", "--name-only", "-z", commit, head, "--", *compared)
    return ModuleDrift(
        directory=module["directory"],
        module=module["module"],
        compared=compared,
        changed=[path for path in changed.split("\0") if path != ""],
        not_compared="",
    )


def measure_drift(repo: pathlib.Path, reading: EntrypointReading) -> ImageDrift:
    """Compare every module a run starts with the submitter's HEAD.

    Args:
        repo: The project's repository, whose history holds the commit its
            images are built from.
        reading: What the run's environment said about its modules.

    Returns:
        The image's commit, the HEAD it was compared with, and per module
        what was compared and what changed. For a run with no image both
        commits are empty and every module says why it was not compared.

    Raises:
        AppError: With ``ENTRYPOINT_IMAGE_COMMIT_UNKNOWN`` when the
            repository does not hold the image's commit, or
            ``ENTRYPOINT_DRIFT_UNREADABLE`` when git cannot answer.
    """
    commit = reading["image_commit"]
    if commit is None:
        return ImageDrift(
            image_commit="",
            head="",
            modules=[
                ModuleDrift(
                    directory=module["directory"],
                    module=module["module"],
                    compared=[],
                    changed=[],
                    not_compared="the run has no image, so no build commit identifies its code",
                )
                for module in reading["modules"]
            ],
        )
    known = _test_hooks.run(["git", "-C", str(repo), "cat-file", "-e", f"{commit}^{{commit}}"])
    if known["returncode"] != 0:
        raise AppError(
            Hpc3ErrorCode.ENTRYPOINT_IMAGE_COMMIT_UNKNOWN,
            f"the image was built from {commit}, which {repo} does not hold: git cat-file -e "
            f"exited {known['returncode']} with {known['stderr'].strip()!r}. So its code cannot "
            "be compared with the submitter's. Fetch it, or submit from the repository that "
            "built the image.",
        )
    head = _git(repo, "rev-parse", "--verify", "HEAD^{commit}").strip()
    if head == "":
        # Refused at write time because the ledger refuses it at read time:
        # a row naming an image commit and no HEAD would make the whole
        # ledger undecodable for every command that reads it.
        raise AppError(
            Hpc3ErrorCode.ENTRYPOINT_DRIFT_UNREADABLE,
            f"git rev-parse in {repo} exited 0 and named no HEAD commit, so there is "
            "nothing to compare the image's code with.",
        )
    # Listed at the resolved commit, not at the name HEAD, so the paths
    # compared are exactly the tree `head` records.
    tracked = _git(repo, "ls-tree", "-r", "-z", "--name-only", head).split("\0")
    return ImageDrift(
        image_commit=commit,
        head=head,
        modules=[
            _module_drift(module, repo=repo, commit=commit, head=head, tracked=tracked)
            for module in reading["modules"]
        ],
    )


def drift_for_command(drift: ImageDrift, command: str, *, working_directory: str) -> ImageDrift:
    """Narrow a submission's drift to the modules one command starts.

    An array's members share one environment and one comparison, but each
    member's ledger row should name only what that member runs.

    Args:
        drift: The submission's drift.
        command: One member's command.
        working_directory: The job's working directory.

    Returns:
        The same commits, with only the invocations ``command`` makes.
    """
    pairs = invocations(command, working_directory=working_directory)
    return ImageDrift(
        image_commit=drift["image_commit"],
        head=drift["head"],
        modules=[
            module
            for module in drift["modules"]
            if (module["directory"], module["module"]) in pairs
        ],
    )


def merge_drifts(drifts: Sequence[ImageDrift]) -> ImageDrift:
    """Combine an array's per-member records into the submission's one.

    Every member of one submission shares its image and its comparison, so
    a command reports the submission once rather than once per member.

    Args:
        drifts: The members' records, at least one, all of one submission.

    Returns:
        The first record's commits, with every invocation any member makes,
        each once, in first-seen order.
    """
    modules: list[ModuleDrift] = []
    for record in drifts:
        modules.extend(module for module in record["modules"] if module not in modules)
    return ImageDrift(
        image_commit=drifts[0]["image_commit"], head=drifts[0]["head"], modules=modules
    )


def describe_drift(drift: ImageDrift) -> list[str]:
    """Render a drift record as the lines a command prints.

    Args:
        drift: The record.

    Returns:
        A line naming the image's commit and the HEAD it was compared with,
        then one line per module. A module whose source changed is a line
        beginning ``WARNING``.
    """
    if drift["image_commit"] == "":
        lines = ["  code: no image, so no build commit to compare"]
    else:
        lines = [
            f"  code: image built from {drift['image_commit']}, compared with HEAD {drift['head']}"
        ]
    if drift["modules"] == []:
        lines.append("  code: the command starts no python -m module, so no source was compared")
    for module in drift["modules"]:
        if module["not_compared"] != "":
            lines.append(f"  {module['module']}: not compared: {module['not_compared']}")
            continue
        under = ", ".join(module["compared"])
        if module["changed"] == []:
            lines.append(f"  {module['module']}: unchanged since the image's commit under {under}")
            continue
        named = ", ".join(module["changed"][:WARNING_PATHS])
        rest = len(module["changed"]) - WARNING_PATHS
        more = f" and {rest} more" if rest > 0 else ""
        lines.append(
            f"  WARNING {module['module']}: {len(module['changed'])} file(s) under {under} "
            "changed since the image's commit, so this run executes the image's older "
            f"code, not HEAD's: {named}{more}"
        )
    return lines


__all__ = [
    "WARNING_PATHS",
    "describe_drift",
    "drift_for_command",
    "measure_drift",
    "merge_drifts",
    "source_paths",
]
