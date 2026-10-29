"""Real process and WebSocket transport twins, including timeout and exit."""

from __future__ import annotations

import socket
import sys
import threading
from collections.abc import Generator
from contextlib import contextmanager

import pytest
from platform_core.json_utils import (
    JSONObject,
    dump_json_str,
    load_json_str,
    narrow_json_to_dict,
    require_dict,
)
from websockets.sync.client import connect
from websockets.sync.server import ServerConnection, serve

from fleet.core.native_rpc import _request, attempt_read_threads, read_threads

SESSION = "11111111-1111-4111-8111-111111111111"


class RuntimeServer:
    """A real WebSocket endpoint speaking the read-only Codex protocol subset."""

    def __init__(self, mode: str = "ok") -> None:
        self.mode = mode
        self.requests: list[JSONObject] = []

    def __call__(self, connection: ServerConnection) -> None:
        for raw in connection:
            assert isinstance(raw, str)
            request = narrow_json_to_dict(load_json_str(raw))
            self.requests.append(request)
            if "id" not in request:
                continue
            if self.mode == "binary":
                connection.send(b"not text")
                continue
            if self.mode == "mismatch":
                connection.send(dump_json_str({"id": 999, "result": {}}))
                continue
            if self.mode == "bool-id":
                connection.send(dump_json_str({"id": False, "result": {}}))
                continue
            if self.mode == "error":
                connection.send(dump_json_str({"id": request["id"], "error": {"code": -1}}))
                continue
            if self.mode == "missing" and request["method"] == "thread/read":
                connection.send(dump_json_str({"id": request["id"], "error": {"code": -32600}}))
                continue
            count = 32 if self.mode == "flood" else 1
            for _ in range(count):
                connection.send(dump_json_str({"method": "thread/status/changed", "params": {}}))
            result: JSONObject = {}
            if request["method"] == "thread/read":
                params = require_dict(request, "params")
                assert params == {"threadId": SESSION, "includeTurns": False}
                result = {
                    "thread": {"id": SESSION, "status": {"type": "active", "activeFlags": []}}
                }
            connection.send(dump_json_str({"id": request["id"], "result": result}))


@contextmanager
def runtime(handler: RuntimeServer) -> Generator[tuple[str, ...], None, None]:
    """Run the actual WebSocket server and return a real proxy child command."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        address: tuple[str, int] = listener.getsockname()
        port = address[1]
        with serve(handler, sock=listener, ping_interval=None, close_timeout=1) as server:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                yield (sys.executable, "-m", "tests.native_proxy_child", str(port))
            finally:
                server.shutdown()
                thread.join(timeout=5)
                assert thread.is_alive() is False


def test_real_proxy_reads_metadata_without_starting_or_resuming_turns() -> None:
    handler = RuntimeServer()
    with runtime(handler) as argv:
        threads, failure = attempt_read_threads(argv, (SESSION,), timeout_seconds=5)
    assert failure is None
    assert threads == ({"id": SESSION, "status": {"type": "active", "activeFlags": []}},)
    assert handler.requests == [
        {
            "id": 0,
            "method": "initialize",
            "params": {
                "clientInfo": {"name": "corvis_fleet_observer", "version": "1.0"},
                "capabilities": {"experimentalApi": True},
            },
        },
        {"method": "initialized"},
        {"id": 1, "method": "thread/read", "params": {"threadId": SESSION, "includeTurns": False}},
    ]


@pytest.mark.parametrize(
    ("mode", "message"),
    [
        ("binary", "expected a text"),
        ("mismatch", "identity mismatch"),
        ("bool-id", "identity mismatch"),
        ("error", "returned an error"),
        ("flood", "notification limit"),
    ],
)
def test_real_protocol_faults_propagate_without_idle(mode: str, message: str) -> None:
    with runtime(RuntimeServer(mode)) as argv, pytest.raises(ValueError, match=message):
        read_threads(argv, (SESSION,), timeout_seconds=5)


def test_real_child_timeout_and_nonzero_exit_report_unavailable() -> None:
    timed_out = attempt_read_threads(
        [sys.executable, "-c", "import time; time.sleep(30)"],
        (SESSION,),
        timeout_seconds=1,
    )
    assert timed_out == ((), "CODEX_OBSERVE_UNAVAILABLE: TimeoutError")
    failed = attempt_read_threads(
        [sys.executable, "-c", "raise SystemExit(7)"],
        (SESSION,),
        timeout_seconds=5,
    )
    assert failed == ((), "CODEX_OBSERVE_UNAVAILABLE: InvalidMessage")


def test_absent_proxy_executable_reports_unavailable() -> None:
    result = attempt_read_threads(
        ["/a-path-that-does-not-exist/codex"],
        (SESSION,),
        timeout_seconds=1,
    )
    assert result == ((), "CODEX_OBSERVE_UNAVAILABLE: FileNotFoundError")


def test_missing_thread_is_an_explicit_refusal_not_an_absent_response() -> None:
    with runtime(RuntimeServer("missing")) as argv:
        result = read_threads(argv, (SESSION,), timeout_seconds=5)
    assert result == ("CODEX_OBSERVE_RPC: thread/read returned an error",)


def test_request_checks_shared_deadline_before_reading() -> None:
    with serve(RuntimeServer(), "127.0.0.1", 0, close_timeout=1) as server:
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            address: tuple[str, int] = server.socket.getsockname()
            port = address[1]
            with (
                connect(f"ws://127.0.0.1:{port}/", proxy=None, close_timeout=1) as connection,
                pytest.raises(TimeoutError, match="CODEX_OBSERVE_TIMEOUT"),
            ):
                _request(connection, 0, "initialize", {}, 0)
        finally:
            server.shutdown()
            thread.join(timeout=5)
            assert thread.is_alive() is False
