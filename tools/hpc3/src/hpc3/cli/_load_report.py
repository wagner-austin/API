"""Post a cluster program's measured reading on its board task, as the program.

A count a session copies into a note is a restatement; these measuring
programs publish their own readings under task-derived service identities.
WRDS uses ``load-report-<task prefix>`` and HPC3 uses ``hpc-report-<task prefix>``.
The shared transport keeps identity registration and publication identical.

The identity is minted with :func:`platform_core.board.service_identity`: the
same label and session id on every run, which the board's one-session,
one-label binding requires of a writer with no harness session.
"""

from __future__ import annotations

import re
from typing import Final, Literal

from platform_core.board import (
    BoardIdentity,
    post_to_task,
    register_service_session,
    service_identity,
)

from hpc3.cli import _test_hooks as cli_hooks
from hpc3.cli._config import taskboard_credentials

_TASK_ID: Final = re.compile(r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")

#: The ledger's record of what kind of writer this is.
HARNESS: Final = "hpc3-wrds-inventory"


def report_identity(task_id: str, family: Literal["wrds", "hpc"]) -> BoardIdentity:
    """Derive the service identity that reports on one task.

    Args:
        task_id: The board task's full id.
        family: Cluster reporting identity; WRDS identities remain stable.

    Returns:
        Label ``load-report-<id[:8]>``, a session id derived from it, and the
        ``service://`` location the board records.

    Raises:
        ValueError: ``WRDS_REPORT_TASK`` unless the id is a full lowercase
            UUID; a prefix could name a different task than the label does.
    """
    if _TASK_ID.fullmatch(task_id) is None:
        raise ValueError(f"WRDS_REPORT_TASK: expected a full task id, got {task_id!r}")
    prefix = "load-report" if family == "wrds" else "hpc-report"
    service = "wrds-inventory" if family == "wrds" else "hpc-evidence"
    label = f"{prefix}-{task_id[:8]}"
    return service_identity(
        agent=label, service=f"corvis:hpc3:{service}:{label}", cwd=f"service://hpc3-{service}"
    )


def post_load_report(task_id: str, body: str, family: Literal["wrds", "hpc"]) -> None:
    """Register the reporting identity and post one report on the task.

    Args:
        task_id: The board task's full id.
        body: The report text.
        family: WRDS inventory or HPC3 execution evidence identity.

    Raises:
        ValueError: From :func:`report_identity`.
        AppError: When the stack's credentials are missing or the board
            refuses either write. Nothing is caught: an unposted report must
            fail the command that measured it.
    """
    identity = report_identity(task_id, family)
    credentials = taskboard_credentials()
    register_service_session(
        cli_hooks.http_post,
        credentials,
        identity,
        harness=identity["cwd"].removeprefix("service://"),
        purpose=(
            "posts the WRDS inventories' measured counts and experiment attempts' "
            "measured outcomes on their task"
            if family == "wrds"
            else "posts native HPC3 accounting, verified artifacts and replay comparisons"
        ),
    )
    post_to_task(
        cli_hooks.http_post, credentials, identity, task_id=task_id, kind="note", body=body
    )


__all__ = ["HARNESS", "post_load_report", "report_identity"]
