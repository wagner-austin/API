"""The hub's pass that ends every runner Worker whose job GitHub dropped.

MCPs board task 715b9ec5, A2. The host's reaper ends a Worker only past the
host's job timeout (:mod:`fleet.core.runner_reaper_render`, rule 3), because
the host alone cannot tell a long job from one GitHub let go of; on
2026-10-16 six such Workers held lavender-wsl's CI slice for 1 h 45 min of a
six-hour timeout, and every dispatch to the node was refused
NODE_OWNER_RESERVED while they did. The hub can tell, from GitHub's runner
list (:mod:`fleet.core.runner_dropped_check`), so each hub tick runs this pass
(:mod:`fleet.cli.hub_passes`):

1. Read every roster host's runners against GitHub.
2. When a host names a dropped Worker, read it again
   :data:`CONFIRM_SECONDS` later, once for every host that named one: a
   runner GitHub reads offline for one reading may be a listener
   reconnecting, which a job outlives.
3. End each Worker both readings name with the same pid and start, through
   the reaper's ``--dropped UNIT PID START`` on the host as root, which
   writes each pid's argv before it kills the Worker's tree and leaves the
   listener running. The reaper refuses a pid that is no longer that Worker,
   so a Worker that ended between the reading and the reap is said, never
   mistaken for another process.

A host ssh cannot reach, or whose helper fails, is a line of the pass, and
the next tick reads it again.

THE ROSTER IS READ AT THE ROLLED REF, as the publish pass reads the registry
(:mod:`fleet.core.declared`): the hub agent runs from the rolled commit and
reaps by that commit's roster, never the checkout's, and reading it from the
agent rather than having the launcher hand it down keeps a launcher that runs
from the checkout, which no roll updates, from disagreeing with the agent
about a flag (:func:`fleet.core.rolled.rolled_state` gives the same reason).
"""

from __future__ import annotations

import pathlib
from typing import Final

from fleet.contracts.runners import HostRunnerSpec, RunnerSpec, parse_runner_spec
from fleet.core import _test_hooks, commit_tree, declared, remote, rolled
from fleet.core.runner_dropped_check import (
    DroppedReading,
    DroppedWorker,
    describe_dropped,
    read_dropped,
)
from fleet.core.runner_reaper_render import REAPER_PATH
from fleet.core.script_values import scriptable

#: The roster, as a path in the API repository.
ROSTER_PATH: Final = "tools/fleet/runners.json"

#: Detail prefix when the roster could not be read at the rolled commit.
ROSTER_READ_FAILED_CODE: Final = "FLEET_ROSTER_READ_FAILED"

#: Seconds between a host's two readings. Two hub ticks are three minutes
#: apart; two minutes inside one tick outlasts a listener's reconnect, whose
#: missed polls GitHub reads as offline, and still ends a dropped job's hold
#: on the slice within the tick that first saw it.
CONFIRM_SECONDS = 120


def read_rolled_roster(checkout: pathlib.Path) -> RunnerSpec | str:
    """Read the roster at the commit :data:`fleet.core.rolled.ROLLED_REF` names now.

    Args:
        checkout: A directory inside the API checkout whose object store
            holds the ref: the agent's records directory.

    Returns:
        The roster, or a ``CODE: message`` refusal when the ref names no
        commit or the file could not be read there; nothing is reaped then.

    Raises:
        AppError: ``RUNNER_SPEC_UNREADABLE`` when the file at the commit does
            not decode as a roster: the rolled roster is then broken, which
            is raised rather than reaped around.
    """
    resolved = commit_tree.resolve_commit(checkout, rolled.ROLLED_REF, rolled.REF_UNRESOLVED_CODE)
    if isinstance(resolved, str):
        return resolved
    commit = resolved["commit"]
    batch = _test_hooks.run(
        ("git", "-C", str(checkout), "cat-file", "--batch"),
        timeout_seconds=commit_tree.TREE_STEP_TIMEOUT_SECONDS,
        stdin_bytes=f"{commit}:{ROSTER_PATH}\n".encode(),
    )
    if batch["returncode"] != 0:
        return commit_tree.step_refusal(
            ROSTER_READ_FAILED_CODE, f"git cat-file --batch {commit}", batch
        )
    # The seam hands stdout back decoded; the roster is UTF-8 JSON, so
    # encoding it again restores the bytes the blob size counts.
    blobs = declared.split_batch(batch["stdout"].encode("utf-8"), [ROSTER_PATH])
    if isinstance(blobs, str):
        return blobs
    return parse_runner_spec(blobs[0], source=f"{commit}:{ROSTER_PATH}")


def _key(held: DroppedWorker) -> tuple[str, int, int]:
    """What names one Worker across two readings.

    Args:
        held: The Worker.

    Returns:
        Its unit, pid and start: a pid alone could name a later process.
    """
    return held["install"]["service"], held["worker"]["pid"], held["worker"]["start"]


def _reap_argv(spec: HostRunnerSpec, held: DroppedWorker) -> tuple[str, ...]:
    """The host command that ends one Worker's tree.

    Args:
        spec: The host.
        held: The Worker.

    Returns:
        The reaper's ``--dropped`` mode, run as root.

    Raises:
        ValueError: When a value cannot be embedded verbatim.
    """
    return (
        *remote.ROOT_PREFIX,
        REAPER_PATH,
        "--dropped",
        scriptable(held["install"]["service"], label="service"),
        str(held["worker"]["pid"]),
        str(held["worker"]["start"]),
    )


def _unread(spec: HostRunnerSpec, reading: DroppedReading) -> str | None:
    """The line for a reading that gave nothing, if it did.

    Args:
        spec: The host.
        reading: Its reading.

    Returns:
        The line, or None when the host answered.
    """
    failure = reading["failure"]
    if failure is None:
        return None
    return f"{spec['name']}: the dropped-job reading gave nothing: {failure['message']}"


def _reap(spec: HostRunnerSpec, confirmed: list[DroppedWorker]) -> list[str]:
    """End each confirmed Worker, one line per Worker and per reaper line.

    Args:
        spec: The host.
        confirmed: The Workers both readings named.

    Returns:
        What the reaper wrote for each, or why it could not be run.
    """
    lines: list[str] = []
    for held in confirmed:
        lines.append(f"{spec['name']}: ending {describe_dropped(held)}")
        outcome = remote.attempt_ssh(spec["host"], _reap_argv(spec, held))
        failure = outcome["failure"]
        if failure is not None:
            lines.append(f"{spec['name']}: the reaper did not end it: {failure['message']}")
            continue
        lines.extend(f"{spec['name']}: {line}" for line in outcome["output"].splitlines())
    return lines


def reap_pass(runners: RunnerSpec) -> list[str]:
    """Read every host, read again where a Worker was named, and end what both name.

    Args:
        runners: The roster.

    Returns:
        The pass's lines, one or more per host.

    Raises:
        AppError: ``RUNNER_GITHUB_UNREADABLE`` and ``RUNNER_AUDIT_UNPARSABLE``
            as :func:`fleet.core.runner_dropped_check.read_dropped` says: a
            reading the hub cannot make is this machine's fault, and ends the
            tick rather than reaping on half of it.
    """
    lines: list[str] = []
    named: list[tuple[HostRunnerSpec, DroppedReading]] = []
    for spec in runners["hosts"]:
        reading = read_dropped(spec)
        unread = _unread(spec, reading)
        if unread is not None:
            lines.append(unread)
        elif reading["dropped"]:
            lines.append(
                f"{spec['name']}: {len(reading['dropped'])} Runner.Worker(s) read as holding a job "
                f"GitHub dropped, read again in {CONFIRM_SECONDS} s: "
                + "; ".join(describe_dropped(held) for held in reading["dropped"])
            )
            named.append((spec, reading))
        else:
            lines.append(
                f"{spec['name']}: examined {reading['examined']} runner(s); none holds a "
                "job GitHub dropped"
            )
    if not named:
        return lines
    _test_hooks.sleep(CONFIRM_SECONDS)
    for spec, first in named:
        again = read_dropped(spec)
        unread = _unread(spec, again)
        if unread is not None:
            lines.append(unread)
            continue
        before = {_key(held) for held in first["dropped"]}
        confirmed = [held for held in again["dropped"] if _key(held) in before]
        left = len(first["dropped"]) - len(confirmed)
        if left:
            lines.append(f"{spec['name']}: {left} Runner.Worker(s) no longer read as dropped, left")
        lines.extend(_reap(spec, confirmed))
    return lines


__all__ = [
    "CONFIRM_SECONDS",
    "ROSTER_PATH",
    "ROSTER_READ_FAILED_CODE",
    "read_rolled_roster",
    "reap_pass",
]
