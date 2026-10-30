"""Join registered host identities to native Codex runtime observations.

The board remains the registry and writer. Native status is evidence with a
short expiry, never a replacement claim based on a daemon PID or board age.
Only already registered, self-declared Codex sessions on this host are read.
"""

from __future__ import annotations

from datetime import UTC, datetime

from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    load_json_str,
    narrow_json_to_dict,
    require_bool,
    require_dict,
    require_int,
    require_list,
    require_str,
)
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.core import _test_hooks, _test_hooks_native
from fleet.core.ssh_options import SSH_OPTIONS


def registered_codex(board: McpCredentials, machine: str, identity: JSONObject) -> tuple[str, ...]:
    """Read every page of the existing ledger's machine-readable inventory.

    Args:
        board: Existing taskboard credentials.
        machine: Exact machine name measured by the host observation.
        identity: Observer identity for the existing tool call.

    Returns:
        Unique registered self-declared Codex thread identities on that host.

    Raises:
        ValueError: Host identity or pagination contradicts the request.
        AppError: Transport or JSON contract failure.
    """
    offset = 0
    targets: set[str] = set()
    while True:
        document = narrow_json_to_dict(
            load_json_str(
                call_mcp_tool(
                    _test_hooks.http_post,
                    board,
                    "task_sessions_list",
                    {
                        **identity,
                        "machine": machine,
                        "format": "json",
                        "limit": 200,
                        "offset": offset,
                    },
                )
            )
        )
        rows = require_list(document, "rows")
        for value in rows:
            row = narrow_json_to_dict(value)
            if require_str(row, "machine") != machine:
                raise ValueError("CODEX_OBSERVE_IDENTITY: ledger returned another machine")
            harness = require_str(row, "harness")
            session = require_str(row, "sessionId")
            declared = require_bool(row, "selfRegistered")
            if harness == "codex" and declared:
                targets.add(session)
        page = require_dict(document, "pagination")
        if require_int(page, "offset") != offset or require_int(page, "limit") != 200:
            raise ValueError("CODEX_OBSERVE_PAGE: ledger returned another page window")
        if not require_bool(page, "hasMore"):
            return tuple(sorted(targets))
        next_offset = require_int(page, "nextOffset")
        if next_offset != offset + len(rows) or not rows:
            raise ValueError("CODEX_OBSERVE_PAGE: ledger pagination did not advance")
        offset = next_offset


def thread_status(thread: JSONObject | str, session_id: str) -> tuple[str, str | None]:
    """Validate the runtime identity and translate its explicit status.

    Args:
        thread: Native thread/read result with turns excluded.
        session_id: Registered thread that was requested.

    Returns:
        Board native status and optional runtime error detail.

    Raises:
        ValueError: Unknown status, flags, or mismatched thread identity.
        AppError: Malformed JSON fields.
    """
    if isinstance(thread, str):
        return "unavailable", thread
    if require_str(thread, "id") != session_id:
        raise ValueError("CODEX_OBSERVE_IDENTITY: runtime returned another thread")
    status = require_dict(thread, "status")
    kind = require_str(status, "type")
    if kind == "systemError":
        return "system-error", "Codex runtime reports systemError"
    if kind == "idle":
        return "idle", None
    if kind == "notLoaded":
        return "not-loaded", None
    if kind != "active":
        raise ValueError("CODEX_OBSERVE_STATUS: unknown runtime status")
    flags = require_list(status, "activeFlags")
    if any(flag not in ("waitingOnApproval", "waitingOnUserInput") for flag in flags):
        raise ValueError("CODEX_OBSERVE_STATUS: unknown active flag")
    if "waitingOnApproval" in flags:
        return "waiting-approval", None
    if "waitingOnUserInput" in flags:
        return "waiting-input", None
    return "active", None


def observe_codex(
    board: McpCredentials,
    host: str,
    user: str,
    machine: str,
    identity: JSONObject,
    evidence_seconds: int,
) -> list[JSONValue]:
    """Produce bounded native evidence through the existing SSH route.

    Args:
        board: Existing taskboard credentials.
        host: SSH alias from the fleet registry.
        user: Provisioned account running Codex.
        machine: Host's measured ledger key.
        identity: Existing fleet observer identity.
        evidence_seconds: The workspace's ``native_evidence_seconds``, how
            long each observation stays evidence: past the gap between two
            observe passes, which runs longer than one tick's period.

    Returns:
        Native observations for task_session_observe. A failed runtime read
        explicitly produces unavailable evidence; an empty inventory does not
        invoke a proxy. Each expires ``evidence_seconds`` after it was read.

    Raises:
        ValueError: A runtime response does not match registered identities.
        AppError: Registry or native JSON contract failure.
    """
    targets = registered_codex(board, machine, identity)
    if not targets:
        return []
    if len(targets) > 200:
        raise ValueError("CODEX_OBSERVE_LIMIT: more than 200 registered Codex sessions on host")
    observed = _test_hooks.now()
    threads, failure = _test_hooks_native.read(
        ["ssh", *SSH_OPTIONS, host, f"/home/{user}/.local/bin/codex", "app-server", "proxy"],
        targets,
        timeout_seconds=20,
    )
    if failure is None and len(threads) != len(targets):
        raise ValueError("CODEX_OBSERVE_RESPONSE: incomplete runtime response batch")
    statuses = (
        [thread_status(thread, session) for thread, session in zip(threads, targets, strict=True)]
        if failure is None
        else [("unavailable", failure) for _ in targets]
    )
    return [
        {
            "sessionId": session,
            "harness": "codex",
            "status": status,
            "detail": detail,
            "observedAt": datetime.fromtimestamp(observed, UTC).isoformat(),
            "expiresAt": datetime.fromtimestamp(observed + evidence_seconds, UTC).isoformat(),
        }
        for session, (status, detail) in zip(targets, statuses, strict=True)
    ]


def evidence_note(observations: list[JSONValue], evidence_seconds: int) -> str:
    """The clause the observe pass's log line carries for its native evidence.

    MCPs execution's served observe-pass-gaps case reads it from the hub's
    day logs and holds each gap between two passes to the lifetime the
    earlier one stamped (MCPs board task 17f74771): the ledger keeps one row
    per session, so only the log records every pass.

    Args:
        observations: What :func:`observe_codex` produced for one pass.
        evidence_seconds: The lifetime each was stamped with.

    Returns:
        ``""`` for a pass with none, else ``; native evidence for <n> Codex
        session(s) read at <instant> lasts <seconds> s``.

    Raises:
        JSONTypeError: An observation without its ``observedAt``.
    """
    if not observations:
        return ""
    read_at = require_str(narrow_json_to_dict(observations[0]), "observedAt")
    return (
        f"; native evidence for {len(observations)} Codex session(s) read at {read_at} "
        f"lasts {evidence_seconds} s"
    )


__all__ = ["evidence_note", "observe_codex", "registered_codex", "thread_status"]
