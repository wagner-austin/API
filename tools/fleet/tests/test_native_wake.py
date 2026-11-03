"""The hub's half of waking an idle Codex owner: stages, evidence and failures.

MCPs board task 562cfa7a, acceptance A1. The session host's ``codex-wake.py``
answers through the ssh seam with the lines its own suites prove it prints
(MCPs ``packages/claude-hooks/codex-wake.test.py``), the native transport is
the seam whose real twin is ``tests/test_native_turns.py``, and the board is
the queue fake that speaks the real JSON-RPC shape. Asserted: what is run on
the host, in what order, what is posted, and that a failure at any stage
leaves the wake pending rather than recorded.
"""

from __future__ import annotations

from collections.abc import Generator, Sequence
from typing import Literal

import pytest
from platform_core.json_utils import JSONTypeError, JSONValue, dump_json_str

from fleet.core import _test_hooks, _test_hooks_native, native_wake, observe
from fleet.core.native_turns import WakeDelivery, WakeRequest
from fleet.core.native_wake_contract import PreparedWake, decode_prepare
from fleet.core.ssh_options import SSH_OPTIONS
from tests._queue_fakes import FakeQueue
from tests.conftest import FakeRun, failed, ok
from tests.test_native_observe import Runtime, identity_row, inventory
from tests.test_observe import BOARD, EVIDENCE, IDENTITY, document, node

SESSION = "11111111-1111-4111-8111-111111111111"
KEY = "22222222-2222-4222-8222-222222222222"
SCRIPT = "/home/corvis/.claude/hooks/codex-wake.py"
PROXY = ("ssh", *SSH_OPTIONS, "ziggurat", "/home/corvis/.local/bin/codex", "app-server", "proxy")


def ssh(*words: str) -> tuple[str, ...]:
    """The argv the hub runs for one codex-wake verb."""
    return ("ssh", *SSH_OPTIONS, "ziggurat", "python3", SCRIPT, *words)


def answer(*, new: bool = True, wake: bool = True, open_turns: Sequence[str] = ()) -> str:
    """A prepare line, as codex-wake.py's encode_answer prints it."""
    pending: JSONValue = (
        {
            "key": KEY,
            "agent": "codex-owner-1103",
            "new": new,
            "prepared_at": "2026-11-03T09:30:00+00:00",
            "events": [
                {
                    "at": "2026-11-03T09:29:00.000Z",
                    "kind": "note",
                    "author": "fleet",
                    "task": "6c501b6a",
                },
                {"at": "2026-11-03T09:29:01.000Z", "kind": "note", "author": "peer", "task": ""},
            ],
            "message_ids": [3093],
            "text": "BOARD WAKE: a fleet job passed",
        }
        if wake
        else None
    )
    opened: list[JSONValue] = [
        {"key": f"k-{turn}", "turn_id": turn, "agent": "codex-owner-1103"} for turn in open_turns
    ]
    return dump_json_str({"wake": pending, "open": opened}) + "\n"


class Deliver:
    """The transport seam, answering one scripted delivery and recording each request."""

    def __init__(self, delivery: WakeDelivery | None, failure: str | None = None) -> None:
        self.delivery = delivery
        self.failure = failure
        self.calls: list[tuple[tuple[str, ...], str, WakeRequest | None, tuple[str, ...], int]] = []

    def __call__(
        self,
        argv: Sequence[str],
        thread_id: str,
        wake: WakeRequest | None,
        open_turns: tuple[str, ...],
        *,
        timeout_seconds: int,
    ) -> tuple[WakeDelivery | None, str | None]:
        self.calls.append((tuple(argv), thread_id, wake, open_turns, timeout_seconds))
        return self.delivery, self.failure


def delivered(
    outcome: Literal["started", "found", "left", "none"],
    turn: str | None,
    ended: dict[str, str] | None = None,
    runtime: str = "idle",
) -> WakeDelivery:
    """A delivery as the transport reports one."""
    return WakeDelivery(runtime=runtime, outcome=outcome, turn_id=turn, ended=ended or {})


@pytest.fixture(autouse=True)
def reset_native_hooks() -> Generator[None, None, None]:
    saved_deliver = _test_hooks_native.deliver
    saved_read = _test_hooks_native.read
    yield
    _test_hooks_native.deliver = saved_deliver
    _test_hooks_native.read = saved_read


def bodies(board: FakeQueue) -> list[JSONValue]:
    """Every task_post body, asserting each went to the standing task as a note."""
    posts = [
        args for tool, args in zip(board.tools, board.arguments, strict=True) if tool == "task_post"
    ]
    for post in posts:
        assert post["taskId"] == native_wake.CODEX_WAKE_TASK
        assert post["kind"] == "note"
        assert post["agent"] == IDENTITY["agent"]
    return [post["body"] for post in posts]


def test_a_new_wake_is_queued_started_committed_and_each_stage_posted() -> None:
    run = FakeRun([ok(answer()), ok('{"stage": "accepted"}\n')])
    _test_hooks.run = run
    board = FakeQueue(["posted", "posted"])
    _test_hooks.http_post = board
    deliver = Deliver(delivered("started", "turn-new"))
    _test_hooks_native.deliver = deliver
    lines = native_wake.wake_session(BOARD, IDENTITY, "ziggurat", "corvis", SESSION)
    assert lines == [f"codex wake {SESSION}: turn turn-new started for key {KEY}"]
    assert run.calls == [
        ssh("prepare", SESSION),
        ssh("commit", SESSION, KEY, "turn-new", "started"),
    ]
    assert deliver.calls == [
        (PROXY, SESSION, {"key": KEY, "text": "BOARD WAKE: a fleet job passed"}, (), 20)
    ]
    assert bodies(board) == [
        f"CODEX WAKE QUEUED codex-owner-1103 thread {SESSION} key {KEY}: 2 row(s) "
        "[2026-11-03T09:29:00.000Z note by fleet on 6c501b6a; "
        "2026-11-03T09:29:01.000Z note by peer on board], message(s) [3093], "
        "selected 2026-11-03T09:30:00+00:00",
        f"CODEX WAKE ACCEPTED codex-owner-1103 thread {SESSION} key {KEY} turn turn-new (started)",
    ]


def test_a_repeated_poll_queues_nothing_again_and_commits_the_turn_it_finds() -> None:
    run = FakeRun([ok(answer(new=False)), ok("{}\n")])
    _test_hooks.run = run
    board = FakeQueue(["posted"])
    _test_hooks.http_post = board
    _test_hooks_native.deliver = Deliver(delivered("found", "turn-earlier", runtime="active"))
    lines = native_wake.wake_session(BOARD, IDENTITY, "ziggurat", "corvis", SESSION)
    assert lines == [f"codex wake {SESSION}: turn turn-earlier found for key {KEY}"]
    assert run.calls[1] == ssh("commit", SESSION, KEY, "turn-earlier", "found")
    assert [str(body).split(" ")[2] for body in bodies(board)] == ["ACCEPTED"]


def test_nothing_pending_runs_prepare_only() -> None:
    run = FakeRun([ok(answer(wake=False))])
    _test_hooks.run = run
    deliver = Deliver(None, "never asked")
    _test_hooks_native.deliver = deliver
    assert native_wake.wake_session(BOARD, IDENTITY, "ziggurat", "corvis", SESSION) == []
    assert deliver.calls == []


def test_a_host_that_cannot_prepare_is_a_line_and_nothing_else() -> None:
    _test_hooks.run = FakeRun([failed(1, "CODEX_WAKE_UNBOUND: no label")])
    deliver = Deliver(None, "never asked")
    _test_hooks_native.deliver = deliver
    (line,) = native_wake.wake_session(BOARD, IDENTITY, "ziggurat", "corvis", SESSION)
    assert line.startswith(f"codex wake {SESSION}: prepare failed: ")
    assert "CODEX_WAKE_UNBOUND" in line
    assert deliver.calls == []


@pytest.mark.parametrize("wake", [True, False])
def test_a_transport_failure_leaves_the_wake_pending_uncommitted(wake: bool) -> None:
    run = FakeRun([ok(answer(new=False, wake=wake, open_turns=("turn-open",)))])
    _test_hooks.run = run
    _test_hooks_native.deliver = Deliver(None, "CODEX_WAKE_UNAVAILABLE: TimeoutError")
    lines = native_wake.wake_session(BOARD, IDENTITY, "ziggurat", "corvis", SESSION)
    pending = f"key {KEY}" if wake else "nothing"
    assert lines == [
        f"codex wake {SESSION}: {pending} left pending: CODEX_WAKE_UNAVAILABLE: TimeoutError"
    ]
    assert run.calls == [ssh("prepare", SESSION)]


def test_a_thread_busy_by_the_time_it_is_read_is_left_pending() -> None:
    run = FakeRun([ok(answer(new=False))])
    _test_hooks.run = run
    _test_hooks_native.deliver = Deliver(
        delivered("left", None, runtime="active:waitingOnApproval")
    )
    lines = native_wake.wake_session(BOARD, IDENTITY, "ziggurat", "corvis", SESSION)
    assert lines == [
        f"codex wake {SESSION}: key {KEY} left pending: thread active:waitingOnApproval"
    ]
    assert run.calls == [ssh("prepare", SESSION)]


def test_a_failed_commit_says_the_next_pass_finds_the_turn() -> None:
    _test_hooks.run = FakeRun([ok(answer(new=False)), failed(1, "CODEX_WAKE_POSITION: x")])
    board = FakeQueue([])
    _test_hooks.http_post = board
    _test_hooks_native.deliver = Deliver(delivered("started", "turn-new"))
    (line,) = native_wake.wake_session(BOARD, IDENTITY, "ziggurat", "corvis", SESSION)
    assert line.startswith(
        f"codex wake {SESSION}: turn turn-new started for key {KEY}, commit failed "
        "(the next pass finds the turn by its key): "
    )
    assert bodies(board) == []


def test_an_open_turn_that_ended_is_completed_and_posted_and_a_failed_record_said() -> None:
    run = FakeRun(
        [
            ok(answer(wake=False, open_turns=("turn-a", "turn-b", "turn-c"))),
            ok("{}\n"),
            failed(1, "CODEX_WAKE_MISMATCH: x"),
        ]
    )
    _test_hooks.run = run
    board = FakeQueue(["posted"])
    _test_hooks.http_post = board
    deliver = Deliver(delivered("none", None, {"turn-a": "completed", "turn-c": "failed"}))
    _test_hooks_native.deliver = deliver
    lines = native_wake.wake_session(BOARD, IDENTITY, "ziggurat", "corvis", SESSION)
    assert deliver.calls == [(PROXY, SESSION, None, ("turn-a", "turn-b", "turn-c"), 20)]
    assert run.calls[1:] == [
        ssh("complete", SESSION, "k-turn-a", "turn-a", "completed"),
        ssh("complete", SESSION, "k-turn-c", "turn-c", "failed"),
    ]
    assert lines[0] == f"codex wake {SESSION} turn turn-a ended completed"
    assert lines[1].startswith(f"codex wake {SESSION} turn turn-c ended failed, not recorded: ")
    assert bodies(board) == [
        f"CODEX WAKE COMPLETED codex-owner-1103 thread {SESSION} key k-turn-a turn turn-a "
        "status completed"
    ]


def test_only_idle_observations_are_woken() -> None:
    run = FakeRun([ok(answer(wake=False))])
    _test_hooks.run = run
    _test_hooks_native.deliver = Deliver(None, "never asked")
    observations: list[JSONValue] = [
        {"sessionId": "busy", "status": "active"},
        {"sessionId": SESSION, "status": "idle"},
        {"sessionId": "waiting", "status": "waiting-approval"},
    ]
    assert native_wake.wake_idle(BOARD, IDENTITY, "ziggurat", "corvis", observations) == []
    assert run.calls == [ssh("prepare", SESSION)]


def test_the_observe_pass_appends_each_wake_to_its_node_line() -> None:
    _test_hooks.run = FakeRun([ok(""), ok(document(hostname="ziggurat")), ok(answer()), ok("{}\n")])
    _test_hooks.now = lambda: 1000
    _test_hooks.http_post = FakeQueue(
        [inventory(identity_row()), "observed native evidence", "posted", "posted"]
    )
    _test_hooks_native.read = Runtime(({"id": SESSION, "status": {"type": "idle"}},))
    _test_hooks_native.deliver = Deliver(delivered("started", "turn-new"))
    result = observe.observe_node(BOARD, node("ziggurat", "session-host"), IDENTITY, EVIDENCE)
    assert result["detail"].endswith(
        f"observed native evidence; codex wake {SESSION}: turn turn-new started for key {KEY}"
    )


def test_the_prepare_line_decodes_and_refuses_another_programs_line() -> None:
    decoded = decode_prepare(answer(open_turns=("t",)))
    assert decoded["open"] == [{"key": "k-t", "turn_id": "t", "agent": "codex-owner-1103"}]
    wake = decoded["wake"]
    assert (wake or {"message_ids": []})["message_ids"] == [3093]
    assert decode_prepare(answer(wake=False)) == {"wake": None, "open": []}
    with pytest.raises(JSONTypeError, match="'wake' is missing"):
        decode_prepare('{"open": []}')
    with pytest.raises(JSONTypeError):
        decode_prepare(dump_json_str({"wake": {"key": KEY}, "open": []}))


def test_the_script_is_the_installed_hooks_tree() -> None:
    assert native_wake.wake_script("corvis") == SCRIPT


def test_a_queued_note_with_nothing_carried_says_none() -> None:
    empty = PreparedWake(
        key=KEY,
        agent="codex-owner-1103",
        new=True,
        prepared_at="2026-11-03T09:30:00+00:00",
        events=[],
        message_ids=[],
        text="BOARD WAKE",
    )
    note = native_wake.queued_note(SESSION, empty)
    assert "0 row(s) [none], message(s) [none]" in note
    assert "@" not in note
