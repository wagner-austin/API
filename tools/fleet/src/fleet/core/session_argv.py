"""The one session-audit invocation each session verb runs.

Split from :mod:`fleet.core.restart` by concern (MCPs board task 553cc92f):
that module decides whether a session job may run and maps its verb to an
invocation; this one composes the argv itself, one function per verb, every
one through :func:`session_audit_argv`, so each verb's line is the release's
own session-audit on the session's host. That session-audit reads the board
register its own release carries, which MCPs alone locates
(``session_audit.registry_paths``): no path to it is passed from here, since
the copy this module once passed outlived the register's move (MCPs board
task ebd671be). Every value that reaches an argv element here was
grammar-checked by :mod:`fleet.core.restart` first.
"""

from __future__ import annotations

import pathlib
from typing import Final

#: Where session-audit lives inside a release's checkout.
SESSION_AUDIT_DIR: Final = pathlib.PurePosixPath("packages/session-audit")


def session_audit_argv(release_root: pathlib.Path, host: str, *arguments: str) -> tuple[str, ...]:
    """Compose one session-audit invocation from a release.

    Args:
        release_root: The verified release's checkout
            (:class:`~fleet.core.session_release.SessionRelease`), whose
            sealed session-audit environment the invocation runs in.
        host: The node the target session runs on, the claiming tick's
            ``--node``, which session-audit acts on.
        arguments: The mode and its arguments.

    Returns:
        ``poetry -C <release>/packages/session-audit run session-audit
        <arguments> --host <node>``. Through poetry, the same way the
        supervisor's runner invokes the package, so the invocation resolves
        the release's own in-project venv, whose editable install is the
        release's source, rather than this runner's; that install's default
        register is the release's own, so the window a kill re-checks and the
        room a handoff joins are the sealed ones.
    """
    return (
        "poetry",
        "-C",
        str(release_root / pathlib.Path(SESSION_AUDIT_DIR)),
        "run",
        "session-audit",
        *arguments,
        "--host",
        host,
    )


def restart_argv(release_root: pathlib.Path, host: str, session_target: str) -> tuple[str, ...]:
    """Compose the session-audit invocation for one restart job.

    Args:
        release_root: The verified release's checkout.
        host: The node the session runs on.
        session_target: The session UUID (already grammar-checked).

    Returns:
        The argv: ``session-audit rollover --apply --session <uuid>``
        through :func:`session_audit_argv`.
    """
    return session_audit_argv(
        release_root, host, "rollover", "--apply", "--session", session_target
    )


def revive_argv(
    release_root: pathlib.Path, host: str, session_target: str, requested_by: str
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one revive job.

    Args:
        release_root: The verified release's checkout.
        host: The node the session runs on.
        session_target: The session UUID (already grammar-checked).
        requested_by: The label that enqueued the revive (already
            grammar-checked), named in the brief session-audit types.

    Returns:
        The argv: ``session-audit revive --session <uuid> --requested-by
        <label>`` through :func:`session_audit_argv`.
    """
    return session_audit_argv(
        release_root, host, "revive", "--session", session_target, "--requested-by", requested_by
    )


def kill_argv(
    release_root: pathlib.Path,
    host: str,
    session_target: str,
    requested_by: str,
    *,
    hard: bool,
    exiting: bool = False,
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one kill job.

    Args:
        release_root: The verified release's checkout, whose register
            carries the idle window the graceful kill re-checks its premise
            against.
        host: The node the session runs on.
        session_target: The session UUID (already grammar-checked).
        requested_by: The label that enqueued the kill (already
            grammar-checked), printed with session-audit's outcome.
        hard: Whether the queue row is ``kill-session-hard``.
        exiting: Whether the queue row is ``exit-session``, a session's own
            exit its supervisor approved, which a session already gone has
            met (MCPs board task 7829e947); a kill otherwise.

    Returns:
        The argv: ``session-audit kill --session <uuid> --requested-by
        <label>`` through :func:`session_audit_argv`, with ``--hard`` after
        the label for the hard verb and ``--exit`` for an exit, and neither
        otherwise.
    """
    base = ("kill", "--session", session_target, "--requested-by", requested_by)
    hard_flag: tuple[str, ...] = ("--hard",) if hard else ()
    exit_flag: tuple[str, ...] = ("--exit",) if exiting else ()
    return session_audit_argv(release_root, host, *base, *hard_flag, *exit_flag)


def compact_argv(
    release_root: pathlib.Path, host: str, session_target: str, requested_by: str
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one compact job.

    Args:
        release_root: The verified release's checkout.
        host: The node the session runs on.
        session_target: The session UUID (already grammar-checked).
        requested_by: The label that enqueued the compact (already
            grammar-checked), printed with session-audit's outcome.

    Returns:
        The argv: ``session-audit compact --session <uuid> --requested-by
        <label>`` through :func:`session_audit_argv`.
    """
    return session_audit_argv(
        release_root, host, "compact", "--session", session_target, "--requested-by", requested_by
    )


def move_argv(
    release_root: pathlib.Path, host: str, session_target: str, requested_by: str
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one move job.

    Args:
        release_root: The verified release's checkout.
        host: The node the session runs on.
        session_target: The session UUID (already grammar-checked).
        requested_by: The label that enqueued the move (already
            grammar-checked), named in the line session-audit types into the
            moved session.

    Returns:
        The argv: ``session-audit move --session <uuid> --requested-by
        <label>`` through :func:`session_audit_argv`.
    """
    return session_audit_argv(
        release_root, host, "move", "--session", session_target, "--requested-by", requested_by
    )


def handoff_argv(
    release_root: pathlib.Path,
    host: str,
    session_target: str,
    task_id: str,
    requested_by: str,
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one handoff job.

    Args:
        release_root: The verified release's checkout, whose register's
            supervision table names the model and slots of the room the
            fresh worker joins.
        host: The node the predecessor session runs on, where its tree is
            and the fresh worker starts.
        session_target: The predecessor's session UUID (already
            grammar-checked).
        task_id: The board task the fresh worker starts on (already
            grammar-checked).
        requested_by: The label that handed the task over (already
            grammar-checked), named in the fresh worker's brief.

    Returns:
        The argv: ``session-audit handoff --session <uuid> --task <uuid>
        --requested-by <label>`` through :func:`session_audit_argv`.
    """
    return session_audit_argv(
        release_root,
        host,
        "handoff",
        "--session",
        session_target,
        "--task",
        task_id,
        "--requested-by",
        requested_by,
    )


__all__ = [
    "SESSION_AUDIT_DIR",
    "compact_argv",
    "handoff_argv",
    "kill_argv",
    "move_argv",
    "restart_argv",
    "revive_argv",
    "session_audit_argv",
]
