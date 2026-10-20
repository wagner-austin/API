"""The one session-audit invocation each session verb runs.

Split from :mod:`fleet.core.restart` by concern (MCPs board task 553cc92f):
that module decides whether a session job may run and maps its verb to an
invocation; this one composes the argv itself, one function per verb, every
one through :func:`session_audit_argv`, so each verb's line is the release's
own session-audit against the release's own register on the session's host.
Every value that reaches an argv element here was grammar-checked by
:mod:`fleet.core.restart` first.
"""

from __future__ import annotations

import pathlib
from typing import Final

#: Where session-audit lives inside a release's checkout.
SESSION_AUDIT_DIR: Final = pathlib.PurePosixPath("packages/session-audit")


def session_audit_argv(
    release_root: pathlib.Path, registry_dir: str, host: str, *arguments: str
) -> tuple[str, ...]:
    """Compose one session-audit invocation against a release's register.

    Args:
        release_root: The verified release's checkout
            (:class:`~fleet.core.session_release.SessionRelease`), whose
            sealed session-audit environment the invocation runs in.
        registry_dir: The release's register, so the window a kill
            re-checks is the sealed one.
        host: The node the target session runs on, the claiming tick's
            ``--node``, which session-audit acts on.
        arguments: The mode and its arguments.

    Returns:
        ``poetry -C <release>/packages/session-audit run session-audit
        <arguments> --host <node> --registry-dir <dir>``. Through poetry,
        the same way the supervisor's runner invokes the package, so the
        invocation resolves the release's own in-project venv, whose
        editable install is the release's source, rather than this runner's.
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
        "--registry-dir",
        registry_dir,
    )


def restart_argv(
    release_root: pathlib.Path, registry_dir: str, host: str, session_target: str
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one restart job.

    Args:
        release_root: The verified release's checkout.
        registry_dir: The release's register.
        host: The node the session runs on.
        session_target: The session UUID (already grammar-checked).

    Returns:
        The argv: ``session-audit rollover --apply --session <uuid>``
        through :func:`session_audit_argv`.
    """
    return session_audit_argv(
        release_root, registry_dir, host, "rollover", "--apply", "--session", session_target
    )


def revive_argv(
    release_root: pathlib.Path,
    registry_dir: str,
    host: str,
    session_target: str,
    requested_by: str,
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one revive job.

    Args:
        release_root: The verified release's checkout.
        registry_dir: The release's register.
        host: The node the session runs on.
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
        host,
        "revive",
        "--session",
        session_target,
        "--requested-by",
        requested_by,
    )


def kill_argv(
    release_root: pathlib.Path,
    registry_dir: str,
    host: str,
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
        host: The node the session runs on.
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
    return session_audit_argv(
        release_root, registry_dir, host, *base, *(("--hard",) if hard else ())
    )


def compact_argv(
    release_root: pathlib.Path,
    registry_dir: str,
    host: str,
    session_target: str,
    requested_by: str,
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one compact job.

    Args:
        release_root: The verified release's checkout.
        registry_dir: The release's register.
        host: The node the session runs on.
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
        host,
        "compact",
        "--session",
        session_target,
        "--requested-by",
        requested_by,
    )


def move_argv(
    release_root: pathlib.Path,
    registry_dir: str,
    host: str,
    session_target: str,
    requested_by: str,
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one move job.

    Args:
        release_root: The verified release's checkout.
        registry_dir: The release's register.
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
        release_root,
        registry_dir,
        host,
        "move",
        "--session",
        session_target,
        "--requested-by",
        requested_by,
    )


def handoff_argv(
    release_root: pathlib.Path,
    registry_dir: str,
    host: str,
    session_target: str,
    task_id: str,
    requested_by: str,
) -> tuple[str, ...]:
    """Compose the session-audit invocation for one handoff job.

    Args:
        release_root: The verified release's checkout.
        registry_dir: The release's register, whose supervision table names
            the model and slots of the room the fresh worker joins.
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
        registry_dir,
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
