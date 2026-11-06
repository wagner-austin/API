"""The wake protocol against a real WebSocket runtime through the real proxy child.

MCPs board task 562cfa7a, acceptance A1: a thread that is busy or waiting is
left alone, a key a turn already carries is found rather than started again,
and a transport failure is reported as a value so the caller keeps the wake
pending. The server speaks the subset of the Codex protocol the wake uses,
read from 0.166.0's schema; every method and field the wake sends is also in
the schema of 0.154.0, the release the session host's app-server runs (MCPs
board task 59b6684d). The client, its process and its socket pair are the
production ones.
"""

from __future__ import annotations

import sys

import pytest
from platform_core.json_utils import (
    JSONObject,
    JSONValue,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
    require_str,
)
from websockets.sync.server import ServerConnection

from fleet.core.native_turns import (
    TURN_PAGE,
    WakeRequest,
    attempt_deliver_wake,
    deliver_wake,
    runtime_word,
)
from tests.test_native_rpc import SESSION, runtime

KEY = "22222222-2222-4222-8222-222222222222"
WAKE = WakeRequest(key=KEY, text="BOARD WAKE: a fleet job passed")


def turn(turn_id: str, status: str, client: str | None) -> JSONObject:
    """One summary turn, its user message carrying ``client`` as its clientId."""
    items: list[JSONValue] = [{"type": "agentMessage", "id": f"{turn_id}-reply", "text": "ok"}]
    items.insert(
        0,
        {
            "type": "userMessage",
            "id": f"{turn_id}-user",
            "clientId": client,
            "content": [{"type": "text", "text": "x", "text_elements": []}],
        },
    )
    return {"id": turn_id, "status": status, "items": items, "itemsView": "summary"}


class TurnServer:
    """A real endpoint answering thread/read, thread/turns/list and turn/start."""

    def __init__(
        self,
        status: JSONObject,
        turns: list[JSONValue],
        *,
        thread_id: str = SESSION,
        refuse_start: bool = False,
    ) -> None:
        self.status = status
        self.turns = turns
        self.thread_id = thread_id
        self.refuse_start = refuse_start
        self.requests: list[JSONObject] = []

    def __call__(self, connection: ServerConnection) -> None:
        for raw in connection:
            assert isinstance(raw, str)
            request = narrow_json_to_dict(load_json_str(raw))
            self.requests.append(request)
            if "id" not in request:
                continue
            method = require_str(request, "method")
            result: JSONObject = {}
            if method == "thread/read":
                result = {"thread": {"id": self.thread_id, "status": self.status}}
            elif method == "thread/turns/list":
                result = {"data": self.turns, "nextCursor": None}
            elif method == "turn/start":
                if self.refuse_start:
                    connection.send(dump_json_str({"id": request["id"], "error": {"code": -32600}}))
                    continue
                result = {"turn": {"id": "turn-new", "status": "inProgress", "items": []}}
            connection.send(dump_json_str({"id": request["id"], "result": result}))

    def methods(self) -> list[str]:
        """Every method requested after the handshake, in order."""
        return [require_str(item, "method") for item in self.requests[2:]]


IDLE: JSONObject = {"type": "idle"}


def test_an_idle_thread_gets_one_turn_carrying_the_key_and_no_setting() -> None:
    server = TurnServer(IDLE, [turn("turn-old", "completed", "operator-typed")])
    with runtime(server) as argv:
        delivery = deliver_wake(argv, SESSION, WAKE, (), timeout_seconds=5)
    assert delivery == {"runtime": "idle", "outcome": "started", "turn_id": "turn-new", "ended": {}}
    assert server.methods() == ["thread/read", "thread/turns/list", "turn/start"]
    assert require_dict(server.requests[3], "params") == {
        "threadId": SESSION,
        "limit": TURN_PAGE,
        "sortDirection": "desc",
        "itemsView": "summary",
    }
    # No model, effort, sandbox, approval or cwd: the thread keeps its own.
    assert require_dict(server.requests[4], "params") == {
        "threadId": SESSION,
        "input": [{"type": "text", "text": WAKE["text"], "text_elements": []}],
        "clientUserMessageId": KEY,
    }
    assert require_dict(require_dict(server.requests[0], "params"), "clientInfo") == {
        "name": "corvis_fleet_wake",
        "version": "1.0",
    }


def test_a_key_a_turn_already_carries_is_found_and_never_started_again() -> None:
    server = TurnServer(
        {"type": "active", "activeFlags": []},
        [turn("turn-mine", "inProgress", KEY), turn("turn-old", "completed", None)],
    )
    with runtime(server) as argv:
        delivery = deliver_wake(argv, SESSION, WAKE, (), timeout_seconds=5)
    assert delivery == {
        "runtime": "active",
        "outcome": "found",
        "turn_id": "turn-mine",
        "ended": {},
    }
    assert "turn/start" not in server.methods()


@pytest.mark.parametrize(
    ("status", "word"),
    [
        ({"type": "active", "activeFlags": []}, "active"),
        ({"type": "active", "activeFlags": ["waitingOnApproval"]}, "active:waitingOnApproval"),
        ({"type": "active", "activeFlags": ["waitingOnUserInput"]}, "active:waitingOnUserInput"),
        ({"type": "notLoaded"}, "notLoaded"),
        ({"type": "systemError"}, "systemError"),
    ],
)
def test_a_thread_that_is_not_idle_is_left_alone(status: JSONObject, word: str) -> None:
    server = TurnServer(status, [])
    with runtime(server) as argv:
        delivery = deliver_wake(argv, SESSION, WAKE, (), timeout_seconds=5)
    assert delivery == {"runtime": word, "outcome": "left", "turn_id": None, "ended": {}}
    assert server.methods() == ["thread/read", "thread/turns/list"]


def test_open_turns_report_only_their_final_ends_without_a_wake() -> None:
    server = TurnServer(
        IDLE,
        [
            turn("turn-a", "completed", "k-a"),
            turn("turn-b", "inProgress", "k-b"),
            turn("turn-c", "interrupted", "k-c"),
            {"id": "turn-d", "status": "failed", "items": [{"type": "reasoning", "id": "r"}]},
        ],
    )
    with runtime(server) as argv:
        delivery = deliver_wake(
            argv,
            SESSION,
            None,
            ("turn-a", "turn-b", "turn-c", "turn-d", "turn-gone"),
            timeout_seconds=5,
        )
    assert delivery == {
        "runtime": "idle",
        "outcome": "none",
        "turn_id": None,
        "ended": {"turn-a": "completed", "turn-c": "interrupted", "turn-d": "failed"},
    }
    assert "turn/start" not in server.methods()


def test_an_answer_for_another_thread_is_a_contract_fault() -> None:
    other = "33333333-3333-4333-8333-333333333333"
    with (
        runtime(TurnServer(IDLE, [], thread_id=other)) as argv,
        pytest.raises(ValueError, match="CODEX_WAKE_IDENTITY"),
    ):
        deliver_wake(argv, SESSION, WAKE, (), timeout_seconds=5)


def test_a_refused_start_is_a_failure_value_so_the_wake_stays_pending() -> None:
    with runtime(TurnServer(IDLE, [], refuse_start=True)) as argv:
        result = attempt_deliver_wake(argv, SESSION, WAKE, (), timeout_seconds=5)
    assert result == (None, "CODEX_WAKE_REFUSED: CODEX_OBSERVE_RPC: turn/start returned an error")


def test_a_real_child_that_hangs_exits_or_is_absent_reports_unavailable() -> None:
    hung = attempt_deliver_wake(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        SESSION,
        WAKE,
        (),
        timeout_seconds=1,
    )
    assert hung == (None, "CODEX_WAKE_UNAVAILABLE: TimeoutError")
    exited = attempt_deliver_wake(
        [sys.executable, "-c", "raise SystemExit(7)"], SESSION, WAKE, (), timeout_seconds=5
    )
    assert exited == (None, "CODEX_WAKE_UNAVAILABLE: InvalidMessage")
    absent = attempt_deliver_wake(
        ["/a-path-that-does-not-exist/codex"], SESSION, WAKE, (), timeout_seconds=1
    )
    assert absent == (None, "CODEX_WAKE_UNAVAILABLE: FileNotFoundError")


def test_runtime_word_names_flags_after_a_colon() -> None:
    assert runtime_word({"type": "idle"}) == "idle"
    assert (
        runtime_word({"type": "active", "activeFlags": ["waitingOnApproval", "waitingOnUserInput"]})
        == "active:waitingOnApproval,waitingOnUserInput"
    )
