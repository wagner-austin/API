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

FIVE SESSION VERBS, ONE TABLE (MCPs migs 507, 525, 526 and 564). Restart,
revive, the two kills (``kill-session`` and ``kill-session-hard``, board
task 660964d9) and the compact (``compact-session``, board task 01f31e4a)
each map to exactly one session-audit invocation through
:func:`session_invocation`, so the runner has one job path for all of them
and no verb can reach an invocation another verb owns. The hard kill is its
own verb on the queue and its own ``--hard`` flag here: nothing in this
module can turn a graceful kill into a hard one, and nothing can turn a
compact, which keeps the session running, into a kill.

EVERY VERB RUNS A SEALED SESSION-VERBS RELEASE (MCPs board task c7c2527d).
The invocation runs the session-audit, the private environment and the
register of the release :mod:`fleet.core.session_release` found and
verified for this job, with no ``PYTHONPATH`` of its own and bytecode writes
off, so nothing the verb does changes the release under its seal. Neither
the MCPs checkout's working tree nor ``origin/main`` is ever what acts, and
the closing detail names the release id and revision.
"""

from __future__ import annotations

import pathlib
import re
from typing import Final, TypedDict

from fleet.contracts.dispatch import DispatchCommand
from fleet.core import _test_hooks
from fleet.core._test_hooks import CommandResult
from fleet.core.rebuild import DETAIL_TAIL_CHARS
from fleet.core.session_release import SessionRelease

#: The UUID grammar the harness writes session ids in, and mig 507 pins.
SESSION_TARGET_PATTERN: Final = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)

#: Where session-audit lives inside a release's checkout.
SESSION_AUDIT_DIR: Final = pathlib.PurePosixPath("packages/session-audit")

#: Detail prefix for a target outside the UUID grammar.
TARGET_INVALID_CODE: Final = "RESTART_TARGET_INVALID"

#: Detail prefix for a restart-session row that carries no target -- a
#: queue that has stopped enforcing its own CHECK, which is worth naming.
TARGET_MISSING_CODE: Final = "RESTART_TARGET_MISSING"

#: The session verbs this runner executes on the hub, each mapped to a
#: session-audit mode by :func:`session_invocation` (MCPs migs 507, 525, 526
#: and 564). ``revive-session`` (board task 1fe89973) is the twin of
#: ``restart-session`` for a session with NO pane and no process:
#: session-audit opens a new tmux window and resumes the transcript there.
#: ``compact-session`` (board task 01f31e4a) shrinks a working session that
#: got too big, the operator's "/compact if they get too big", never an
#: exit. ``exit-session`` (MCPs board task 5aa8ed06) is a session's own exit,
#: approved by its room's supervisor. The words are the queue's own
#: :class:`DispatchCommand` members, so one vocabulary names them.
SESSION_COMMANDS: Final[tuple[DispatchCommand, ...]] = (
    DispatchCommand.RESTART_SESSION,
    DispatchCommand.REVIVE_SESSION,
    DispatchCommand.KILL_SESSION,
    DispatchCommand.KILL_SESSION_HARD,
    DispatchCommand.COMPACT_SESSION,
    DispatchCommand.EXIT_SESSION,
)

#: Detail prefix for a command this module has no invocation for.
COMMAND_UNKNOWN_CODE: Final = "SESSION_COMMAND_UNKNOWN"

#: The board label grammar, for the ``--requested-by`` a revive carries into
#: the brief it types and a kill prints with its outcome. Refused before it
#: reaches an argv element, like the target: a revive's is typed into a
#: terminal by session-audit.
LABEL_PATTERN: Final = re.compile(r"^[a-z0-9][a-z0-9-]{2,63}$")

#: Detail prefix for a revive or kill whose submitter label is not a board label.
REQUESTER_INVALID_CODE: Final = "SESSION_REQUESTER_INVALID"


def refusal_for(session_target: str | None) -> str | None:
    """Decide whether a session job's target can honestly reach an argv element.

    The release the verb runs is judged separately, by
    :func:`fleet.core.session_release.active_session_release`.

    Args:
        session_target: The queue row's target, or None when the row
            carries none.

    Returns:
        A ``CODE: message`` refusal detail for the queue, or None when the
        job can run.
    """
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


def session_audit_argv(
    release_root: pathlib.Path, registry_dir: str, *arguments: str
) -> tuple[str, ...]:
    """Compose one session-audit invocation against a release's register.

    Args:
        release_root: The verified release's checkout
            (:class:`~fleet.core.session_release.SessionRelease`), whose
            sealed session-audit environment the invocation runs in.
        registry_dir: The release's register, so the window a kill
            re-checks is the sealed one.
        arguments: The mode and its arguments.

    Returns:
        ``poetry -C <release>/packages/session-audit run session-audit
        <arguments> --registry-dir <dir>``. Through poetry, the same way the
        supervisor's runner invokes the package, so the invocation resolves
        the release's own in-project venv, whose editable install is the
        release's source, rather than this runner's.
    """
    return (
        "poetry",
        "-C",
        str(release_root / pathlib.Path(SESSION_AUDIT_DIR)),
        "run",
        "session-audit",
        *arguments,
        "--registry-dir",
        registry_dir,
    )


def restart_argv(
    release_root: pathlib.Path, registry_dir: str, session_target: str
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one restart job.

    Args:
        release_root: The verified release's checkout.
        registry_dir: The release's register.
        session_target: The session UUID (already grammar-checked).

    Returns:
        The argv: ``session-audit rollover --apply --session <uuid>``
        through :func:`session_audit_argv`.
    """
    return session_audit_argv(
        release_root, registry_dir, "rollover", "--apply", "--session", session_target
    )


def revive_argv(
    release_root: pathlib.Path, registry_dir: str, session_target: str, requested_by: str
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one revive job.

    Args:
        release_root: The verified release's checkout.
        registry_dir: The release's register.
        session_target: The session UUID (already grammar-checked).
        requested_by: The label that enqueued the revive (already
            grammar-checked), named in the brief session-audit types.

    Returns:
        The argv: ``session-audit revive --session <uuid> --requested-by
        <label>`` through :func:`session_audit_argv`.
    """
    return session_audit_argv(
        release_root,
        registry_dir,
        "revive",
        "--session",
        session_target,
        "--requested-by",
        requested_by,
    )


def kill_argv(
    release_root: pathlib.Path,
    registry_dir: str,
    session_target: str,
    requested_by: str,
    *,
    hard: bool,
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one kill job.

    Args:
        release_root: The verified release's checkout.
        registry_dir: The release's register, which carries the idle window
            the graceful kill re-checks its premise against.
        session_target: The session UUID (already grammar-checked).
        requested_by: The label that enqueued the kill (already
            grammar-checked), printed with session-audit's outcome.
        hard: Whether the queue row is ``kill-session-hard``.

    Returns:
        The argv: ``session-audit kill --session <uuid> --requested-by
        <label>`` through :func:`session_audit_argv`, with ``--hard`` after
        the label for the hard verb and never otherwise.
    """
    base = ("kill", "--session", session_target, "--requested-by", requested_by)
    return session_audit_argv(release_root, registry_dir, *base, *(("--hard",) if hard else ()))


def compact_argv(
    release_root: pathlib.Path, registry_dir: str, session_target: str, requested_by: str
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one compact job.

    Args:
        release_root: The verified release's checkout.
        registry_dir: The release's register.
        session_target: The session UUID (already grammar-checked).
        requested_by: The label that enqueued the compact (already
            grammar-checked), printed with session-audit's outcome.

    Returns:
        The argv: ``session-audit compact --session <uuid> --requested-by
        <label>`` through :func:`session_audit_argv`.
    """
    return session_audit_argv(
        release_root,
        registry_dir,
        "compact",
        "--session",
        session_target,
        "--requested-by",
        requested_by,
    )


class SessionInvocation(TypedDict):
    """What one session verb runs.

    Attributes:
        verb: The run-id prefix: ``restart``, ``revive``, ``kill`` or
            ``compact``.
        mode: The session-audit mode the closing detail names.
        argv: The one invocation.
        types_requester: Whether the submitter label becomes an argv
            element, and so must be judged before the run.
        release: The id of the sealed release whose session-audit runs,
            named in the closing detail.
        revision: The commit that release was checked out at, named beside
            it.
    """

    verb: str
    mode: str
    argv: tuple[str, ...]
    types_requester: bool
    release: str
    revision: str


def session_invocation(
    release: SessionRelease,
    command: DispatchCommand,
    session_target: str,
    requested_by: str,
) -> SessionInvocation:
    """Map one session verb to its one invocation.

    Args:
        release: The verified release the verb runs
            (:func:`fleet.core.session_release.active_session_release`).
        command: The queue row's command.
        session_target: The session UUID (already grammar-checked).
        requested_by: The queue row's ``submitted_by``.

    Returns:
        The :class:`SessionInvocation`.

    Raises:
        ValueError: ``SESSION_COMMAND_UNKNOWN`` for a command that is not a
            session verb. The caller routes only :data:`SESSION_COMMANDS`
            here, so this names a routing defect rather than guessing.
    """
    root = release["root"]
    registry_dir = release["registry_dir"]
    release_id = release["release"]
    revision = release["revision"]
    if command is DispatchCommand.RESTART_SESSION:
        return SessionInvocation(
            verb="restart",
            mode="rollover",
            argv=restart_argv(root, registry_dir, session_target),
            types_requester=False,
            release=release_id,
            revision=revision,
        )
    if command is DispatchCommand.REVIVE_SESSION:
        return SessionInvocation(
            verb="revive",
            mode="revive",
            argv=revive_argv(root, registry_dir, session_target, requested_by),
            types_requester=True,
            release=release_id,
            revision=revision,
        )
    if command in (DispatchCommand.KILL_SESSION, DispatchCommand.KILL_SESSION_HARD):
        return SessionInvocation(
            verb="kill",
            mode="kill",
            argv=kill_argv(
                root,
                registry_dir,
                session_target,
                requested_by,
                hard=command is DispatchCommand.KILL_SESSION_HARD,
            ),
            types_requester=True,
            release=release_id,
            revision=revision,
        )
    if command is DispatchCommand.COMPACT_SESSION:
        return SessionInvocation(
            verb="compact",
            mode="compact",
            argv=compact_argv(root, registry_dir, session_target, requested_by),
            types_requester=True,
            release=release_id,
            revision=revision,
        )
    if command is DispatchCommand.EXIT_SESSION:
        # The graceful kill's keystrokes, idle re-read and pane close, with
        # nothing of its own: an approved self-exit is the same act, and the
        # queue's verb is what the ledger records it by (MCPs 5aa8ed06).
        return SessionInvocation(
            verb="exit",
            mode="kill",
            argv=kill_argv(root, registry_dir, session_target, requested_by, hard=False),
            types_requester=True,
            release=release_id,
            revision=revision,
        )
    raise ValueError(f"{COMMAND_UNKNOWN_CODE}: {command.value!r} is not a session verb")


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
            "refused before it could become an argv element or be typed into a pane"
        )
    return None


#: Environment variables a session-audit invocation must not inherit from
#: this agent. The agent runs under ``poetry run`` (the scheduled action
#: runs :mod:`fleet.cli.tick` that way), which exports ``VIRTUAL_ENV`` naming the
#: FLEET venv, and poetry prefers an activated venv to the ``-C`` project's
#: own: with it inherited, ``poetry -C packages/session-audit run
#: session-audit`` executed in the fleet venv and died on
#: ``ModuleNotFoundError: No module named 'session_audit'`` (dispatch job
#: f7c3b8a2, 2026-09-17 00:46Z, the first session verb ever run live).
#: Withholding this one variable makes poetry resolve session-audit's own
#: ``.venv`` even with the fleet venv first on PATH (measured both ways).
#: ``PYTHONPATH`` is withheld too: anything on it would be imported ahead of
#: the release's sealed source (MCPs board task c7c2527d).
SESSION_ENVIRONMENT_EXCLUDED: Final[tuple[str, ...]] = ("VIRTUAL_ENV", "PYTHONPATH")

#: What a session verb runs with: no bytecode written, because a
#: ``__pycache__`` written into the release's source or environment would
#: change it under its seal, and the next job's verification would refuse
#: it (MCPs board task c7c2527d).
SESSION_ENVIRONMENT_SET: Final[tuple[tuple[str, str], ...]] = (("PYTHONDONTWRITEBYTECODE", "1"),)

#: A session verb's deadline, in seconds.
#:
#: session-audit's own rails bound each verb (a revive waits 30 seconds for
#: a prompt and then for a record; a kill waits for a pid to leave the
#: table; a compact waits at most six minutes for the context to fall), so
#: a verb that is still running after ten minutes is one whose
#: pane, ssh hop or poetry resolution has stopped answering, and the job
#: closes failed with that fact instead of holding the tick.
SESSION_JOB_TIMEOUT_SECONDS: Final[int] = 600


def run_session_job(invocation: SessionInvocation) -> CommandResult:
    """Run one session verb, blocking until session-audit reports or the deadline passes.

    Args:
        invocation: What :func:`session_invocation` composed.

    Returns:
        The invocation's exit status and captured streams. Exit 0 means the
        verb did what it names (RESTARTED, REVIVED, ENDED, COMPACTED); anything else
        means it did not, and the stdout tail carries session-audit's own
        outcome line saying why. After :const:`SESSION_JOB_TIMEOUT_SECONDS`
        it is the timed-out result.
    """
    return _test_hooks.run(
        invocation["argv"],
        timeout_seconds=SESSION_JOB_TIMEOUT_SECONDS,
        unset_env=SESSION_ENVIRONMENT_EXCLUDED,
        set_env=SESSION_ENVIRONMENT_SET,
    )


def describe_result(result: CommandResult, invocation: SessionInvocation) -> str:
    """Compose a closing detail from a finished session job.

    Args:
        result: The invocation's outcome.
        invocation: What ran: its session-audit mode (``rollover``,
            ``revive``, ``kill`` or ``compact``), named so the reader knows which outcome
            block the tail carries, and the sealed release and revision it
            ran from, so a closure can show its fix was the code that acted.

    Returns:
        The mode, the release, its revision, the exit code and the tail of the combined
        output -- the tail, because session-audit prints its outcome block
        (``ROLLOVER APPLIED`` with one line per session, ``REVIVE -
        <OUTCOME>``, ``KILL - <OUTCOME>`` or ``COMPACT - <OUTCOME>``) last.
    """
    combined = (result["stdout"] + result["stderr"]).strip()
    tail = combined[-DETAIL_TAIL_CHARS:]
    return (
        f"session-audit {invocation['mode']} from sealed release {invocation['release']} "
        f"(revision {invocation['revision']}) exited {result['returncode']}: {tail}"
    )


__all__ = [
    "COMMAND_UNKNOWN_CODE",
    "LABEL_PATTERN",
    "REQUESTER_INVALID_CODE",
    "SESSION_AUDIT_DIR",
    "SESSION_COMMANDS",
    "SESSION_ENVIRONMENT_EXCLUDED",
    "SESSION_ENVIRONMENT_SET",
    "SESSION_TARGET_PATTERN",
    "TARGET_INVALID_CODE",
    "TARGET_MISSING_CODE",
    "SessionInvocation",
    "compact_argv",
    "describe_result",
    "kill_argv",
    "refusal_for",
    "requester_refusal",
    "restart_argv",
    "revive_argv",
    "run_session_job",
    "session_audit_argv",
    "session_invocation",
]
