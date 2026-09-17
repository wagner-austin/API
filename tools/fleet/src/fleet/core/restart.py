"""Executing a ``restart-session`` queue job on the hub itself.

THE SECOND VERB THAT RUNS HERE INSTEAD OF ON A NODE (MCPs mig 507, board
task ccec3417). The Claude Code sessions live on the hub, in tmux panes
reached over a loopback ssh hop, and ending one and resuming it on the
installed build needs three things no container has: the session
process's own environment (its ssh client port, which is how the pane is
found), the WSL tmux server (which types the keystrokes), and the Windows
process table (which says when the old process is gone). So the job is
enqueued from a container -- ``pc_session_restart`` -- and executed here,
by the runner that already runs as the operator.

THE KEYSTROKES ARE NOT HERE. This module composes exactly one invocation,
``session-audit rollover --apply --session <uuid>`` in the MCPs checkout,
and ``session_audit.rollover`` owns the sequence (Ctrl-C, ``/exit``, wait
for the pid to leave the table, ``claude --resume``, wait for the new record)
and every rail around it: never a session that is not idle, never a host
that is not a pane, never a pane nothing resolved. A second copy of that
sequence here would be the fork that drifts first.

SYNCHRONOUS, LIKE A REBUILD. A restart is well under a minute -- measured
2026-09-13: gone in four seconds, back in twelve -- and closing the job in
the same tick means a died runner leaves nothing half-collected: the row
sits ``running`` until its lease lapses and a later tick runs the
idempotent command again, which the target's own rails make safe (a
session already on the installed build and idle is restarted again, which
costs an unsent draft and nothing else).

THE TARGET GRAMMAR GUARD IS LOAD-BEARING HERE TOO. The queue's CHECK
constraint and the writer both refuse anything but a lowercase hex UUID,
and this runner re-checks before the value lands in an argv element,
because the runner is the layer nearest the machine and the one that
cannot assume the other two were consulted.
"""

from __future__ import annotations

import pathlib
import re
from typing import Final

from fleet.core import _test_hooks
from fleet.core._test_hooks import CommandResult
from fleet.core.rebuild import DETAIL_TAIL_CHARS

#: The UUID grammar the harness writes session ids in, and mig 507 pins.
SESSION_TARGET_PATTERN: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

#: Where session-audit lives inside the MCPs checkout.
SESSION_AUDIT_DIR: Final = pathlib.PurePosixPath("packages/session-audit")

#: Detail prefix for a runner started without the MCPs checkout path.
ROOT_MISSING_CODE: Final = "RESTART_ROOT_MISSING"

#: Detail prefix for a target outside the UUID grammar.
TARGET_INVALID_CODE: Final = "RESTART_TARGET_INVALID"

#: Detail prefix for a restart-session row that carries no target -- a
#: queue that has stopped enforcing its own CHECK, which is worth naming.
TARGET_MISSING_CODE: Final = "RESTART_TARGET_MISSING"

#: The two session verbs this runner executes on the hub, and the
#: session-audit mode each maps to (MCPs migs 507 and 525). ``revive-session``
#: (board task 1fe89973) is the twin of ``restart-session`` for a session
#: with NO pane and no process: session-audit opens a new tmux window and
#: resumes the transcript there. One table, one runner path, so the second
#: verb is a row here rather than a second copy of the job.
RESTART_COMMAND: Final = "restart-session"
REVIVE_COMMAND: Final = "revive-session"
SESSION_COMMANDS: Final[tuple[str, ...]] = (RESTART_COMMAND, REVIVE_COMMAND)

#: The board label grammar, for the ``--requested-by`` a revive carries into
#: the brief it types. Refused before it reaches an argv element, like the
#: target: the value is typed into a terminal by session-audit.
LABEL_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")

#: Detail prefix for a revive whose submitter label is not a board label.
REQUESTER_INVALID_CODE: Final = "REVIVE_REQUESTER_INVALID"


def refusal_for(mcps_root: pathlib.Path, session_target: str | None) -> str | None:
    """Decide whether this runner can honestly execute a restart job.

    The absent-flag case is the CLI's to refuse (it owns the flag and its
    name); this function judges what it is given: the checkout and the
    target.

    Args:
        mcps_root: The MCPs checkout path the runner was started with.
        session_target: The queue row's target, or None when the row
            carries none.

    Returns:
        A ``CODE: message`` refusal detail for the queue, or None when the
        job can run.
    """
    if not _test_hooks.directory_exists(mcps_root):
        return f"{ROOT_MISSING_CODE}: --mcps-root {mcps_root} is not a directory on this machine"
    if session_target is None:
        return (
            f"{TARGET_MISSING_CODE}: a restart-session row carries the session it "
            "restarts and this one does not; the queue's CHECK should have refused it"
        )
    if SESSION_TARGET_PATTERN.fullmatch(session_target) is None:
        return (
            f"{TARGET_INVALID_CODE}: session target {session_target!r} is not a "
            "lowercase hex UUID as the harness writes them; refused before it "
            "could reach an argv element"
        )
    return None


def restart_argv(mcps_root: pathlib.Path, session_target: str) -> tuple[str, ...]:
    """Compose the session-audit invocation for one restart job.

    Args:
        mcps_root: The MCPs checkout (already existence-checked).
        session_target: The session UUID (already grammar-checked).

    Returns:
        The argv: ``poetry -C <mcps>/packages/session-audit run session-audit
        rollover --apply --session <uuid>``. Through poetry, the same way the
        manager runner invokes the package, so the invocation resolves the
        package's own venv rather than this runner's.
    """
    return (
        "poetry",
        "-C",
        str(mcps_root / pathlib.Path(SESSION_AUDIT_DIR)),
        "run",
        "session-audit",
        "rollover",
        "--apply",
        "--session",
        session_target,
    )


def revive_argv(mcps_root: pathlib.Path, session_target: str, requested_by: str) -> tuple[str, ...]:
    """Compose the session-audit invocation for one revive job.

    Args:
        mcps_root: The MCPs checkout (already existence-checked).
        session_target: The session UUID (already grammar-checked).
        requested_by: The label that enqueued the revive (already
            grammar-checked), named in the brief session-audit types.

    Returns:
        The argv: ``poetry -C <mcps>/packages/session-audit run session-audit
        revive --session <uuid> --requested-by <label>``.
    """
    return (
        "poetry",
        "-C",
        str(mcps_root / pathlib.Path(SESSION_AUDIT_DIR)),
        "run",
        "session-audit",
        "revive",
        "--session",
        session_target,
        "--requested-by",
        requested_by,
    )


def requester_refusal(requested_by: str) -> str | None:
    """Judge the submitter label a revive will type into a terminal.

    Args:
        requested_by: The queue row's ``submitted_by``.

    Returns:
        A ``CODE: message`` refusal detail, or None when the label is one
        the board grammar admits.
    """
    if LABEL_PATTERN.fullmatch(requested_by) is None:
        return (
            f"{REQUESTER_INVALID_CODE}: submitter {requested_by!r} is not a board label; "
            "refused before it could be typed into a pane"
        )
    return None


def run_session_revive(
    mcps_root: pathlib.Path, *, session_target: str, requested_by: str
) -> CommandResult:
    """Run the revive, blocking until session-audit reports.

    Args:
        mcps_root: The MCPs checkout.
        session_target: The session UUID to revive.
        requested_by: The label that asked.

    Returns:
        The invocation's exit status and captured streams. Exit 0 means the
        session was REVIVED; anything else means it was not, and the stdout
        tail carries session-audit's ``REVIVE - <OUTCOME>`` line saying why.
    """
    return _test_hooks.run(revive_argv(mcps_root, session_target, requested_by))


def run_session_restart(mcps_root: pathlib.Path, *, session_target: str) -> CommandResult:
    """Run the restart, blocking until session-audit reports.

    Args:
        mcps_root: The MCPs checkout.
        session_target: The session UUID to restart.

    Returns:
        The invocation's exit status and captured streams. Exit 0 means the
        named session was RESTARTED; anything else means it was not, and the
        stdout tail carries session-audit's own outcome line saying why.
    """
    return _test_hooks.run(restart_argv(mcps_root, session_target))


def describe_result(result: CommandResult, mode: str = "rollover") -> str:
    """Compose a closing detail from a finished session job.

    Args:
        result: The invocation's outcome.
        mode: The session-audit mode that ran, ``rollover`` or ``revive``,
            named so the reader knows which outcome block the tail carries.

    Returns:
        The exit code and the tail of the combined output -- the tail,
        because session-audit prints its outcome block (``ROLLOVER
        APPLIED`` with one line per session, or ``REVIVE - <OUTCOME>``) last.
    """
    combined = (result["stdout"] + result["stderr"]).strip()
    tail = combined[-DETAIL_TAIL_CHARS:]
    return f"session-audit {mode} exited {result['returncode']}: {tail}"


__all__ = [
    "LABEL_PATTERN",
    "REQUESTER_INVALID_CODE",
    "RESTART_COMMAND",
    "REVIVE_COMMAND",
    "ROOT_MISSING_CODE",
    "SESSION_AUDIT_DIR",
    "SESSION_COMMANDS",
    "SESSION_TARGET_PATTERN",
    "TARGET_INVALID_CODE",
    "TARGET_MISSING_CODE",
    "describe_result",
    "refusal_for",
    "requester_refusal",
    "restart_argv",
    "revive_argv",
    "run_session_restart",
    "run_session_revive",
]
