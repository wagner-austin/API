"""The fleet's HTTP surface: the pure router, the socket shell, main.

The router is exercised as the pure function it is. The socket handler
is driven single-threadedly over a real ``socketpair`` — the request
bytes are written before the handler reads, so no second thread is ever
needed (this package imports no threading; see the coverage note in
``pyproject.toml``). The real spawn/kill hook implementations get live
throwaway children.
"""

from __future__ import annotations

import socket
from collections.abc import Generator, Sequence
from http.server import HTTPServer
from pathlib import Path
from socketserver import BaseServer

import pytest

from rw_bot.harness import _test_hooks
from rw_bot.harness._test_hooks import _kill_tree_impl, _spawn_match_impl
from rw_bot.harness.fleet import FleetError, FleetManager
from rw_bot.harness.fleet_http import (
    FLEET_PORT_DEFAULT,
    FleetRequestHandler,
    FleetServer,
    main,
    render_json,
    resolve_port,
    route_fleet_request,
)
from tests.test_fleet import FakeSpawner


def _swallow_line(text: str) -> None:
    """Drop one output line — tests assert on state, not chatter.

    Args:
        text: Ignored.
    """


@pytest.fixture()
def spawner() -> Generator[FakeSpawner, None, None]:
    """Install a recording spawner and a silent output line."""
    original_spawn = _test_hooks.spawn_match
    original_write = _test_hooks.write_line
    fake = FakeSpawner()
    _test_hooks.spawn_match = fake
    _test_hooks.write_line = _swallow_line
    yield fake
    _test_hooks.spawn_match = original_spawn
    _test_hooks.write_line = original_write


def _route(
    manager: FleetManager, method: str, path: str, body: bytes = b""
) -> tuple[int, str, bytes]:
    """Route one request.

    Args:
        manager: Manager under test.
        method: HTTP method.
        path: Request path.
        body: Request body.

    Returns:
        ``(status, content type, payload)``.
    """
    return route_fleet_request(manager, method, path, body)


def test_router_serves_the_page_and_the_list(spawner: FakeSpawner) -> None:
    """GET / is the page; GET /bots is the JSON registry."""
    manager = FleetManager()

    status, content_type, payload = _route(manager, "GET", "/")
    assert status == 200
    assert content_type.startswith("text/html")
    assert b"rusted warfare fleet" in payload

    status, content_type, payload = _route(manager, "GET", "/bots")
    assert status == 200
    assert content_type == "application/json"
    assert payload.startswith(b"{")
    assert b'"bots": []' in payload


def test_router_spawn_cycle(spawner: FakeSpawner) -> None:
    """Spawn, list, stop, restart, remove — the whole lifecycle over bytes."""
    manager = FleetManager()
    body = b'{"instance": "alpha", "seed": 7, "fastforward": 8}'

    status, _, payload = _route(manager, "POST", "/bots", body)
    assert status == 201
    assert b'"instance": "alpha"' in payload
    assert spawner.argvs[0][2] == "PLAY_SEED=7"

    killed: list[int] = []

    def fake_kill(pid: int) -> None:
        killed.append(pid)

    original_kill = _test_hooks.kill_tree
    _test_hooks.kill_tree = fake_kill
    try:
        status, _, _ = _route(manager, "POST", "/bots/alpha/stop")
    finally:
        _test_hooks.kill_tree = original_kill
    assert status == 200
    assert killed == [4001]

    status, _, _ = _route(manager, "POST", "/bots/alpha/restart")
    assert status == 409
    status, _, _ = _route(manager, "DELETE", "/bots/alpha")
    assert status == 409

    spawner.matches[0].returncode = 0
    status, _, _ = _route(manager, "POST", "/bots/alpha/restart")
    assert status == 201
    spawner.matches[1].returncode = 0
    status, _, _ = _route(manager, "DELETE", "/bots/alpha")
    assert status == 200


def test_router_stats_and_refusals(spawner: FakeSpawner) -> None:
    """Stats answer JSON; malformed spawns are 400; ghosts are 404."""
    manager = FleetManager()
    _route(manager, "POST", "/bots", b'{"instance": "alpha"}')

    def fake_read(path: Path) -> tuple[str, ...]:
        return ("verdict        A (victory)",)

    original_read = _test_hooks.read_text_lines
    _test_hooks.read_text_lines = fake_read
    try:
        status, content_type, payload = _route(manager, "GET", "/bots/alpha/stats")
    finally:
        _test_hooks.read_text_lines = original_read
    assert status == 200
    assert content_type == "application/json"
    assert b'"finished": true' in payload
    assert b"victory" in payload

    assert _route(manager, "GET", "/bots/ghost/stats")[0] == 404
    assert _route(manager, "POST", "/bots", b"not json")[0] == 400
    assert _route(manager, "POST", "/bots", b'{"instance": ""}')[0] == 400
    assert _route(manager, "POST", "/bots", b'{"instance": "alpha"}')[0] == 409
    assert _route(manager, "POST", "/bots", b'{"instance": "x", "seed": "7"}')[0] == 400
    assert _route(manager, "POST", "/bots", b'{"instance": "x", "seed": true}')[0] == 400
    assert _route(manager, "POST", "/bots", b'{"instance": "x", "map": 3}')[0] == 400
    assert _route(manager, "POST", "/bots/ghost/stop")[0] == 404
    assert _route(manager, "PATCH", "/bots")[0] == 404
    assert _route(manager, "GET", "/nowhere")[0] == 404


def test_render_json_covers_the_response_grammar() -> None:
    """Scalars, escapes, string lists, and row lists all render."""
    rows: list[dict[str, str | int | bool | None]] = [{"pid": 5, "instance": "x"}]
    text = render_json(
        {
            "name": 'a"b\\c\nd\x01',
            "count": 3,
            "on": True,
            "off": False,
            "gone": None,
            "lines": ["one", "two"],
            "rows": rows,
        }
    )
    assert text == (
        '{"name": "a\\"b\\\\c\\nd\\u0001", "count": 3, "on": true,'
        ' "off": false, "gone": null, "lines": ["one", "two"],'
        ' "rows": [{"pid": 5, "instance": "x"}]}'
    )


def _drive_handler(server: HTTPServer, request_bytes: bytes) -> bytes:
    """Run one request through the real handler, single-threadedly.

    The request bytes are fully written into one end of a socketpair
    before the handler ever reads from the other end, so the blocking
    reads inside ``BaseHTTPRequestHandler`` complete immediately.

    Args:
        server: The bound server carrying configuration.
        request_bytes: A complete HTTP/1.0 request.

    Returns:
        The raw response bytes.
    """
    client, service = socket.socketpair()
    try:
        client.sendall(request_bytes)
        client.shutdown(socket.SHUT_WR)
        FleetRequestHandler(service, ("127.0.0.1", 0), server)
        # The server normally closes the connection after the handler
        # runs; here the test plays the server, so it closes explicitly
        # — the response already sits in the client's receive buffer.
        service.close()
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        client.close()


def test_handler_serves_over_a_real_socket(spawner: FakeSpawner) -> None:
    """GET, POST and DELETE all travel the stdlib socket shell."""
    server = FleetServer(("127.0.0.1", 0), FleetManager())
    try:
        response = _drive_handler(server, b"GET /bots HTTP/1.0\r\n\r\n")
        assert response.startswith(b"HTTP/1.0 200")
        assert b'"bots": []' in response

        body = b'{"instance": "alpha"}'
        request = (
            b"POST /bots HTTP/1.0\r\n"
            + f"Content-Length: {len(body)}\r\n\r\n".encode("ascii")
            + body
        )
        response = _drive_handler(server, request)
        assert response.startswith(b"HTTP/1.0 201")

        spawner.matches[0].returncode = 0
        response = _drive_handler(server, b"DELETE /bots/alpha HTTP/1.0\r\n\r\n")
        assert response.startswith(b"HTTP/1.0 200")
    finally:
        server.server_close()


def test_handler_refuses_a_non_fleet_server(spawner: FakeSpawner) -> None:
    """Binding the handler to a plain server is a loud wiring bug."""
    server = HTTPServer(("127.0.0.1", 0), FleetRequestHandler)
    try:
        with pytest.raises(FleetError) as wiring:
            _drive_handler(server, b"GET /bots HTTP/1.0\r\n\r\n")
        assert wiring.value.code == "RW-FLEET-009"
    finally:
        server.server_close()


class _RecordingServer(BaseServer):
    """A server whose accept loop records instead of blocking."""

    def __init__(self) -> None:
        """Bind nothing."""
        super().__init__(("127.0.0.1", 0), FleetRequestHandler)
        self.served = False

    def serve_forever(self, poll_interval: float = 0.5) -> None:
        """Record the call.

        Args:
            poll_interval: Unused.
        """
        self.served = True


def test_real_serve_forever_drives_the_accept_loop() -> None:
    """The production hook delegates to the server's own loop."""
    recorder = _RecordingServer()
    _test_hooks._serve_forever_impl(recorder)
    assert recorder.served is True


def test_main_binds_loopback_and_serves(spawner: FakeSpawner) -> None:
    """main reads --port, binds, announces, and enters the accept loop."""
    served: list[BaseServer] = []
    lines: list[str] = []

    def fake_argv() -> list[str]:
        return ["--port", "0"]

    def fake_serve(server: BaseServer) -> None:
        served.append(server)

    def record_line(text: str) -> None:
        lines.append(text)

    original_argv = _test_hooks.read_argv
    original_serve = _test_hooks.serve_forever
    original_write = _test_hooks.write_line
    _test_hooks.read_argv = fake_argv
    _test_hooks.serve_forever = fake_serve
    _test_hooks.write_line = record_line
    try:
        assert main() == 0
    finally:
        _test_hooks.read_argv = original_argv
        _test_hooks.serve_forever = original_serve
        _test_hooks.write_line = original_write

    if len(served) != 1:
        raise AssertionError(f"expected one serve call, got {served!r}")
    assert any("rw-fleet listening" in line for line in lines)


def test_main_default_port_constant() -> None:
    """The default port stays clear of PLAY_PORT's 27600-27999 range."""
    assert FLEET_PORT_DEFAULT == 27500


def test_resolve_port_contract() -> None:
    """--port overrides; anything else keeps the default."""
    assert resolve_port(["--port", "27501"]) == 27501
    assert resolve_port([]) == FLEET_PORT_DEFAULT
    assert resolve_port(["--port"]) == FLEET_PORT_DEFAULT
    assert resolve_port(["-x", "27501"]) == FLEET_PORT_DEFAULT


def test_real_spawn_and_kill_run_a_live_child(tmp_path: Path) -> None:
    """The production spawn writes the transcript; kill fells the tree.

    The child is a plain Python sleeper — cheap, dependency-free, and
    guaranteed killable. Its transcript proves stdout redirection.
    """
    import sys
    import time

    transcript = tmp_path / "deep" / "child.out"
    argv: Sequence[str] = (
        sys.executable,
        "-c",
        "print('spawned', flush=True); import time; time.sleep(60)",
    )
    child = _spawn_match_impl(argv, transcript)
    assert child.pid > 0
    assert child.poll() is None

    _kill_tree_impl(child.pid)
    deadline = time.monotonic() + 30
    while child.poll() is None and time.monotonic() < deadline:
        time.sleep(0.05)
    if child.poll() is None:
        raise AssertionError("child still running after kill_tree")
    assert transcript.read_text(encoding="utf-8").startswith("spawned")
