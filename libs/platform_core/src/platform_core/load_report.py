"""Post a measuring program's own reading on its board task, as the program.

A count a session copies into a note is a restatement; measuring programs
publish their own readings under task-derived service identities. WRDS
readings (inventories, experiment attempts and the evaluator's verification of
a collected extraction) post as ``load-report-<task prefix>``, HPC3 execution
evidence as ``hpc-report-<task prefix>``.

LIFTED OUT OF ``tools/hpc3`` ON 2026-10-31 (board task 1f60a20d), when the
evaluator-side CRSP verifier in ``libs/covenant_ml`` needed to post under the
same WRDS identity and could not import hpc3. The labels and service names are
the ones hpc3 minted, byte for byte: changing either mints a new identity and
orphans the readings already posted under the old one.

The identity is minted with :func:`platform_core.board.service_identity`: the
same label and session id on every run, which the board's one-session,
one-label binding requires of a writer with no harness session.
"""

from __future__ import annotations

import pathlib
import re
from collections.abc import Callable
from typing import Final, Literal

from platform_core.board import (
    BoardIdentity,
    post_to_task,
    register_service_session,
    service_identity,
)
from platform_core.mcp_client import McpPostProtocol
from platform_core.session_label import taskboard_credentials

_TASK_ID: Final = re.compile(r"\A[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\Z")


def report_identity(task_id: str, family: Literal["wrds", "hpc"]) -> BoardIdentity:
    """Derive the service identity that reports on one task.

    Args:
        task_id: The board task's full id.
        family: Cluster reporting identity; WRDS identities remain stable.

    Returns:
        Label ``load-report-<id[:8]>`` (or ``hpc-report-<id[:8]>``), a session
        id derived from it, and the ``service://`` location the board records.

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


def post_load_report(
    *,
    post: McpPostProtocol,
    read_text: Callable[[pathlib.Path], str],
    task_id: str,
    body: str,
    family: Literal["wrds", "hpc"],
) -> None:
    """Register the reporting identity and post one report on the task.

    Args:
        post: The caller's HTTP seam, from its own ``_test_hooks``.
        read_text: The caller's file seam, for the stack's ``.env`` and its
            endpoint declaration.
        task_id: The board task's full id.
        body: The report text.
        family: WRDS readings or HPC3 execution evidence identity.

    Raises:
        ValueError: From :func:`report_identity`.
        AppError: When the stack's credentials are missing or the board
            refuses either write. Nothing is caught: an unposted report must
            fail the command that measured it.
    """
    identity = report_identity(task_id, family)
    credentials = taskboard_credentials(read_text)
    register_service_session(
        post,
        credentials,
        identity,
        harness=identity["cwd"].removeprefix("service://"),
        purpose=(
            "posts the WRDS inventories' measured counts, experiment attempts' "
            "measured outcomes and evaluator verifications on their task"
            if family == "wrds"
            else "posts native HPC3 accounting, verified artifacts and replay comparisons"
        ),
    )
    post_to_task(post, credentials, identity, task_id=task_id, kind="note", body=body)


__all__ = ["post_load_report", "report_identity"]
