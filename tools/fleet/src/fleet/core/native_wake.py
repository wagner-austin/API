"""Wake idle Codex owners the board addressed, from the observe pass (MCPs task 562cfa7a).

WHY IT EXISTS. A Claude session is woken between turns by its own harness
(MCPs ``packages/claude-hooks/board-wake.py``); Codex has no such hook, so an
addressed event (a fleet job's outcome, a board message) reached an idle
Codex owner only when the operator typed. Measured 2026-11-01: the failure
post abbce245 on task c6bc5d05 did not resume owner 01a17e81 before the
operator's 22:09 request.

WHAT IT REUSES, AND WHAT IT ADDS. The observe pass already reads each
registered Codex thread's runtime status (:mod:`fleet.core.native_observe`).
For each one it read ``idle``, this module:

1. runs ``codex-wake.py prepare`` on the session's host over the existing
   ssh route (:func:`fleet.core.remote.attempt_ssh`). That script selects
   with Claude's own delivery rules and position files, and returns the
   pending wake under a key, plus the wakes whose turns have not been seen
   to end;
2. opens one connection to the host's Codex daemon
   (:func:`fleet.core.native_turns.deliver_wake`), which re-reads the
   thread, reads how open turns ended, and starts the wake's turn only if
   the thread is still idle and no turn already carries the key;
3. runs ``codex-wake.py commit`` once the runtime accepted the turn, and
   ``complete`` for each open turn that ended;
4. posts one note per stage to the standing task :data:`CODEX_WAKE_TASK`:
   QUEUED, ACCEPTED, COMPLETED. The notes name the woken session's label
   without an at-sign, so they address nobody.

FAILURE IS A LINE, NEVER A DROP. A host that does not answer, a daemon that
refuses or a commit that fails each leave the wake pending on the host,
under its key, and say so in the pass's log line; the next pass offers the
same key and finds the turn if it was started. Contract faults (a line or a
response that does not decode) propagate, as the observe pass's do.
"""

from __future__ import annotations

from typing import Final, Literal

from platform_core.json_utils import JSONObject, JSONValue, narrow_json_to_dict, require_str
from platform_core.mcp_client import McpCredentials, call_mcp_tool

from fleet.core import _test_hooks, _test_hooks_native, remote
from fleet.core.native_observe import NATIVE_TIMEOUT_SECONDS, proxy_argv
from fleet.core.native_turns import WakeRequest
from fleet.core.native_wake_contract import OpenDelivery, PreparedWake, decode_prepare

#: The standing task in room ``events`` every stage of every Codex wake is
#: posted to, configuration rather than discovery (MCPs board task 555de7c9).
CODEX_WAKE_TASK: Final = "555de7c9-8487-4a76-8d20-fb5a6ef3bf21"


def wake_script(user: str) -> str:
    """Where the session host's half of the wake is installed.

    Args:
        user: The provisioned account on the host.

    Returns:
        The hooks tree's ``codex-wake.py``, which the MCPs hooks installer
        places on every session host.
    """
    return f"/home/{user}/.claude/hooks/codex-wake.py"


def _post(board: McpCredentials, identity: JSONObject, body: str) -> None:
    """Post one stage note to :data:`CODEX_WAKE_TASK`.

    Args:
        board: The taskboard's credentials.
        identity: This runner's identity arguments.
        body: The note.

    Raises:
        AppError: Transport or contract failure of the board call.
    """
    call_mcp_tool(
        _test_hooks.http_post,
        board,
        "task_post",
        {**identity, "taskId": CODEX_WAKE_TASK, "kind": "note", "body": body},
    )


def queued_note(session: str, wake: PreparedWake) -> str:
    """The QUEUED note: what a wake carries, named by board identity.

    Args:
        session: The Codex thread.
        wake: The wake as prepared.

    Returns:
        The note's body.
    """
    rows = "; ".join(
        f"{event['at']} {event['kind']} by {event['author']} on {event['task'] or 'board'}"
        for event in wake["events"]
    )
    messages = ", ".join(str(item) for item in wake["message_ids"])
    return (
        f"CODEX WAKE QUEUED {wake['agent']} thread {session} key {wake['key']}: "
        f"{len(wake['events'])} row(s) [{rows or 'none'}], "
        f"message(s) [{messages or 'none'}], selected {wake['prepared_at']}"
    )


def _run(host: str, user: str, *words: str) -> tuple[str, str | None]:
    """Run one ``codex-wake.py`` verb on the session host.

    Args:
        host: SSH alias.
        user: The account.
        words: The verb and its arguments.

    Returns:
        Its standard output and no failure, or no output and the failure's
        message.
    """
    outcome = remote.attempt_ssh(host, ("python3", wake_script(user), *words))
    failure = outcome["failure"]
    return (outcome["output"], None) if failure is None else ("", failure["message"])


def _ended(
    board: McpCredentials,
    identity: JSONObject,
    host: str,
    user: str,
    session: str,
    ended: list[tuple[OpenDelivery, str]],
) -> list[str]:
    """Record each open wake whose turn ended.

    Args:
        board: The taskboard's credentials.
        identity: This runner's identity arguments.
        host: SSH alias.
        user: The account.
        session: The Codex thread.
        ended: Each ended delivery with its turn's final status.

    Returns:
        One log clause per turn.
    """
    lines: list[str] = []
    for delivery, status in ended:
        key, turn, agent = delivery["key"], delivery["turn_id"], delivery["agent"]
        _, failure = _run(host, user, "complete", session, key, turn, status)
        if failure is not None:
            lines.append(
                f"codex wake {session} turn {turn} ended {status}, not recorded: {failure}"
            )
            continue
        _post(
            board,
            identity,
            f"CODEX WAKE COMPLETED {agent} thread {session} key {key} turn {turn} status {status}",
        )
        lines.append(f"codex wake {session} turn {turn} ended {status}")
    return lines


def wake_session(
    board: McpCredentials, identity: JSONObject, host: str, user: str, session: str
) -> list[str]:
    """Deliver one idle Codex session's pending wake and record its open turns' ends.

    Args:
        board: The taskboard's credentials.
        identity: This runner's identity arguments.
        host: SSH alias of the session's host.
        user: The provisioned account running Codex there.
        session: The Codex thread, which the runtime read idle this pass.

    Returns:
        Log clauses, none when nothing was pending.

    Raises:
        InvalidJsonError: ``prepare`` printed something that is not JSON.
        JSONTypeError: Its line, or a stage's answer, does not decode.
        AppError: A board post failed.
        ValueError: The runtime answered for another thread.
    """
    output, failure = _run(host, user, "prepare", session)
    if failure is not None:
        return [f"codex wake {session}: prepare failed: {failure}"]
    answer = decode_prepare(output)
    wake = answer["wake"]
    if wake is None and not answer["open"]:
        return []
    if wake is not None and wake["new"]:
        _post(board, identity, queued_note(session, wake))
    delivery, refused = _test_hooks_native.deliver(
        proxy_argv(host, user),
        session,
        None if wake is None else WakeRequest(key=wake["key"], text=wake["text"]),
        tuple(item["turn_id"] for item in answer["open"]),
        timeout_seconds=NATIVE_TIMEOUT_SECONDS,
    )
    if delivery is None:
        pending = "nothing" if wake is None else f"key {wake['key']}"
        return [f"codex wake {session}: {pending} left pending: {refused}"]
    lines = _ended(
        board,
        identity,
        host,
        user,
        session,
        [
            (item, delivery["ended"][item["turn_id"]])
            for item in answer["open"]
            if item["turn_id"] in delivery["ended"]
        ],
    )
    if wake is None:
        return lines
    turn = delivery["turn_id"]
    if turn is None:
        # A wake was requested, so the runtime left it: the thread was not idle.
        return [
            *lines,
            f"codex wake {session}: key {wake['key']} left pending: thread {delivery['runtime']}",
        ]
    accepted: Literal["started", "found"] = (
        "started" if delivery["outcome"] == "started" else "found"
    )
    _, failure = _run(host, user, "commit", session, wake["key"], turn, accepted)
    if failure is not None:
        return [
            *lines,
            f"codex wake {session}: turn {turn} {accepted} for key {wake['key']}, commit failed "
            f"(the next pass finds the turn by its key): {failure}",
        ]
    _post(
        board,
        identity,
        f"CODEX WAKE ACCEPTED {wake['agent']} thread {session} key {wake['key']} "
        f"turn {turn} ({accepted})",
    )
    return [*lines, f"codex wake {session}: turn {turn} {accepted} for key {wake['key']}"]


def wake_idle(
    board: McpCredentials,
    identity: JSONObject,
    host: str,
    user: str,
    observations: list[JSONValue],
) -> list[str]:
    """Run :func:`wake_session` for every Codex session this pass read idle.

    Args:
        board: The taskboard's credentials.
        identity: This runner's identity arguments.
        host: SSH alias.
        user: The account.
        observations: What :func:`fleet.core.native_observe.observe_codex`
            produced this pass.

    Returns:
        Every log clause, in observation order.
    """
    lines: list[str] = []
    for value in observations:
        observation = narrow_json_to_dict(value)
        if require_str(observation, "status") == "idle":
            lines.extend(
                wake_session(board, identity, host, user, require_str(observation, "sessionId"))
            )
    return lines


__all__ = ["CODEX_WAKE_TASK", "queued_note", "wake_idle", "wake_script", "wake_session"]
