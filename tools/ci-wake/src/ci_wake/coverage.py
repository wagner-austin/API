"""Which later run checked a push whose own run never did.

WHY THIS EXISTS. ``wagner-austin/MCPs`` gives every push to main one shared
pending slot (its ``check.yml``, ruled 2026-09-24 on MCPs board task
06362fd0), so of a burst of pushes only the newest waits and every other one
gets a run that is cancelled before any job starts. The run that does
execute narrows from the last GREEN ancestor, so it checks every superseded
push's files. This bridge asked GitHub only about runs whose head IS the
pushed sha, so each superseded push was announced as NOT RUN and the run
that actually checked it was announced to nobody: measured 2026-10-15, 50 of
60 runs on main were cancelled with no job, and five pushes announced NOT
RUN were each inside run 38699479621 (c91c08f97, failure: coloring-book) and
run 38716940740 (e7cb362b9, success). The operator: "howcoke.you csnt just
use their resulr?" (MCPs board task 5af9e749).

WHAT COUNTS AS COVERING, and every clause is a fact the payloads carry, not
a cause this module narrates:
* a COMPLETED run of the SAME WORKFLOW on the SAME BRANCH, created at or
  after the push;
* whose own conclusion is not ``cancelled``, so it is a verdict rather than
  another eviction;
* whose head DESCENDS from the pushed sha, which GitHub's compare answers
  for ``base...head`` with ``ahead`` or ``identical``.
The FIRST such run, by creation, is the push's verdict: it is the earliest
run whose checked tree contains the push. A later green does not overwrite
an earlier red, because the red is what the pusher must act on.

WHAT IT DOES NOT CLAIM. That the covering run's diff window included the
push's files is the repository's policy (narrowing from the last green
ancestor), not something these payloads show; the post names the covering
run and its head so the reader can open it.
"""

from __future__ import annotations

import datetime
from collections.abc import Sequence
from typing import Final
from urllib.parse import quote

from platform_core.json_utils import (
    JSONValue,
    narrow_json_to_dict,
    require_int,
    require_list,
    require_str,
)
from typing_extensions import TypedDict

from ci_wake.runs import CANCELLED, WorkflowRun, decode_run, gh_json, is_terminal

#: The prefix a branch ref carries; a push to anything else (a tag) has no
#: branch for a later run to share.
BRANCH_PREFIX: Final = "refs/heads/"

#: Compare statuses meaning the head descends from the base.
DESCENDS: Final = frozenset({"ahead", "identical"})

#: Most runs one ``actions/runs`` page returns; pages are read until the
#: payload's own ``total_count`` is reached, so the earliest covering run is
#: never the one a page cut off.
_PAGE: Final = 100


class BranchRun(TypedDict):
    """One run on a branch, with the commit it checked and when it was made.

    Attributes:
        run: The run as :func:`ci_wake.runs.decode_run` reads it.
        head_sha: The full sha the run checked.
        created_unix: When GitHub created it, whole seconds since the epoch.
    """

    run: WorkflowRun
    head_sha: str
    created_unix: int


def branch_of(ref: str) -> str:
    """Name the branch a pushed ref updates.

    Args:
        ref: The remote ref, e.g. ``refs/heads/main``.

    Returns:
        The branch, or the empty string for a ref that is not a branch.
    """
    return ref[len(BRANCH_PREFIX) :] if ref.startswith(BRANCH_PREFIX) else ""


def _iso(unix: int) -> str:
    """Render whole epoch seconds as the UTC instant GitHub's filter reads."""
    return datetime.datetime.fromtimestamp(unix, tz=datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def branch_runs_argv(repo: str, branch: str, since_unix: int, page: int) -> tuple[str, ...]:
    """Build the command listing a branch's completed runs since an instant.

    Args:
        repo: ``owner/name``.
        branch: The branch.
        since_unix: The earliest creation time asked about.
        page: The 1-based page.

    Returns:
        The argument vector.
    """
    since = quote(f">={_iso(since_unix)}", safe="")
    return (
        "gh",
        "api",
        f"repos/{repo}/actions/runs?branch={quote(branch, safe='')}&status=completed"
        f"&created={since}&per_page={_PAGE}&page={page}",
    )


def _created_unix(entry: dict[str, JSONValue]) -> int:
    """Read a run's ``created_at`` as whole epoch seconds."""
    created = datetime.datetime.fromisoformat(require_str(entry, "created_at"))
    return int(created.timestamp())


def decode_branch_runs(payload: JSONValue) -> tuple[int, tuple[BranchRun, ...]]:
    """Read one page of a branch's runs.

    Args:
        payload: The decoded ``actions/runs`` response.

    Returns:
        The payload's ``total_count`` and the page's runs.

    Raises:
        JSONTypeError: If the payload or a run is not the documented shape.
    """
    listing = narrow_json_to_dict(payload)
    runs = tuple(
        BranchRun(
            run=decode_run(entry),
            head_sha=require_str(entry, "head_sha"),
            created_unix=_created_unix(entry),
        )
        for entry in (
            narrow_json_to_dict(value) for value in require_list(listing, "workflow_runs")
        )
    )
    return require_int(listing, "total_count"), runs


def branch_runs(repo: str, branch: str, since_unix: int) -> tuple[BranchRun, ...]:
    """Read every completed run of a branch created since an instant.

    Args:
        repo: ``owner/name``.
        branch: The branch.
        since_unix: The earliest creation time asked about.

    Returns:
        Every run, page after page until the payload's count is reached.

    Raises:
        AppError: ``GH_COMMAND_FAILED`` when ``gh`` refused.
        JSONTypeError: When a payload is not the shape this package reads.
    """
    collected: list[BranchRun] = []
    page = 1
    while True:
        total, runs = decode_branch_runs(gh_json(branch_runs_argv(repo, branch, since_unix, page)))
        collected.extend(runs)
        if len(runs) == 0 or len(collected) >= total:
            return tuple(collected)
        page += 1


def compare_argv(repo: str, base: str, head: str) -> tuple[str, ...]:
    """Build the command asking whether ``head`` descends from ``base``.

    Args:
        repo: ``owner/name``.
        base: The pushed sha.
        head: A candidate run's head sha.

    Returns:
        The argument vector, its ``--jq`` keeping only the status, since a
        compare payload otherwise carries every changed file.
    """
    return ("gh", "api", f"repos/{repo}/compare/{base}...{head}", "--jq", "{status: .status}")


def decode_compare_status(payload: JSONValue) -> str:
    """Read the status out of :func:`compare_argv`'s reply.

    Args:
        payload: The decoded ``{"status": ...}`` object.

    Returns:
        ``ahead``, ``identical``, ``behind`` or ``diverged``.

    Raises:
        JSONTypeError: If the reply carries no status.
    """
    return require_str(narrow_json_to_dict(payload), "status")


def candidates(runs: Sequence[BranchRun], workflow: str, pushed_unix: int) -> tuple[BranchRun, ...]:
    """Order the runs that could cover a push, earliest first.

    Pure: the ancestry each still needs is asked by :func:`find_cover`.

    Args:
        runs: A branch's completed runs.
        workflow: The workflow the push's own cancelled run belonged to.
        pushed_unix: When the push was made.

    Returns:
        Its completed, not-cancelled runs of that workflow created at or
        after the push, by creation time.
    """
    return tuple(
        sorted(
            (
                entry
                for entry in runs
                if entry["run"]["workflow"] == workflow
                and is_terminal(entry["run"])
                and entry["run"]["conclusion"] != CANCELLED
                and entry["created_unix"] >= pushed_unix
            ),
            key=_creation_order,
        )
    )


def _creation_order(entry: BranchRun) -> tuple[int, int]:
    """Sort key: creation time, then run id for two runs made in one second."""
    return entry["created_unix"], entry["run"]["run_id"]


def find_cover(
    repo: str, sha: str, workflow: str, pushed_unix: int, runs: Sequence[BranchRun]
) -> BranchRun | None:
    """Find the first run that checked a push its own run did not.

    Args:
        repo: ``owner/name``.
        sha: The pushed sha.
        workflow: The workflow its own run was cancelled in.
        pushed_unix: When it was pushed.
        runs: The branch's completed runs since then.

    Returns:
        The earliest candidate whose head descends from ``sha``, or None
        while no completed run contains it.

    Raises:
        AppError: ``GH_COMMAND_FAILED`` when ``gh`` refused a compare.
        JSONTypeError: When a compare reply is not the shape this reads.
    """
    for entry in candidates(runs, workflow, pushed_unix):
        status = decode_compare_status(gh_json(compare_argv(repo, sha, entry["head_sha"])))
        if status in DESCENDS:
            return entry
    return None


__all__ = [
    "BRANCH_PREFIX",
    "DESCENDS",
    "BranchRun",
    "branch_of",
    "branch_runs",
    "branch_runs_argv",
    "candidates",
    "compare_argv",
    "decode_branch_runs",
    "decode_compare_status",
    "find_cover",
]
