"""The match service's HTTP door, from the pure router to the socket shell.

The router really routes and the queue really queues -- only the database
is a fake -- and the handler test drives the real stdlib shell over a
socketpair, single-threadedly, exactly as the fleet suite proved the
pattern.
"""

from __future__ import annotations

import runpy
import socket
import sys
from http.server import HTTPServer
from socketserver import BaseServer

import pytest
from scripts.match_service import (
    SERVICE_PORT_DEFAULT,
    MatchServiceRequestHandler,
    MatchServiceServer,
    main,
)

from rw_bot.harness import _test_hooks as host_hooks
from rw_bot.service import _test_hooks as service_hooks
from rw_bot.service.http import route_service_request
from rw_bot.service.queue import MatchServiceError
from rw_bot.wire.ndjson import parse_object, render_json
from tests.harness_fakes import FakeHost
from tests.service_fakes import FakeConnection

_JOBS_TEXT = (
    "alpha|12345|doctrines/flame-nocover.doctrine|400\n"
    "alpha|777|doctrines/flame-nocover.doctrine|400"
)


def _submission(name: str = "demo") -> bytes:
    return render_json(
        {
            "name": name,
            "jobs": _JOBS_TEXT,
            "lockstep": 75,
            "map_path": "maps/skirmish/[p2]duel_lake.tmx",
            "difficulty": 2,
            "pin_delta": 3,
            "fast_forward": 10,
        }
    ).encode("utf-8")


def test_healthz_answers() -> None:
    conn = FakeConnection()
    status, _kind, payload = route_service_request(conn, "GET", "/healthz", b"")
    assert status == 200
    assert parse_object(payload.decode("utf-8")) == {"ok": True}


def test_a_posted_batch_queues_and_reports_its_counts() -> None:
    """The whole door: submit over the wire, read the counts back."""
    conn = FakeConnection()
    status, _kind, payload = route_service_request(conn, "POST", "/batches", _submission())
    assert status == 201
    assert parse_object(payload.decode("utf-8")) == {"batch": "demo", "queued": 2, "total": 2}
    status, _kind, payload = route_service_request(conn, "GET", "/batches/demo", b"")
    assert status == 200
    counts = parse_object(payload.decode("utf-8"))
    assert counts == {"batch": "demo", "queued": 2, "running": 0, "done": 0, "failed": 0}


def test_a_resubmitted_batch_reports_zero_newly_queued() -> None:
    conn = FakeConnection()
    route_service_request(conn, "POST", "/batches", _submission())
    status, _kind, payload = route_service_request(conn, "POST", "/batches", _submission())
    assert status == 201
    assert parse_object(payload.decode("utf-8"))["queued"] == 0


def test_a_malformed_job_line_is_a_400_with_the_error_text() -> None:
    conn = FakeConnection()
    body = render_json(
        {
            "name": "demo",
            "jobs": "not a job line",
            "lockstep": 75,
            "map_path": "",
            "difficulty": 0,
            "pin_delta": 0,
            "fast_forward": 0,
        }
    ).encode("utf-8")
    status, _kind, payload = route_service_request(conn, "POST", "/batches", body)
    assert status == 400
    assert b"RW-SWEEP" in payload


def test_a_missing_field_is_a_400() -> None:
    conn = FakeConnection()
    status, _kind, payload = route_service_request(conn, "POST", "/batches", b'{"name":"x"}')
    assert status == 400
    assert b"RW-DECODE" in payload


def test_an_unknown_path_is_a_404() -> None:
    conn = FakeConnection()
    status, _kind, _payload = route_service_request(conn, "GET", "/nothing", b"")
    assert status == 404


def _drive_handler(server: HTTPServer, request_bytes: bytes) -> bytes:
    """Run one request through the real handler, single-threadedly.

    Args:
        server: The bound server carrying the queue connection.
        request_bytes: A complete HTTP/1.0 request.

    Returns:
        The raw response bytes.
    """
    client, service = socket.socketpair()
    try:
        client.sendall(request_bytes)
        client.shutdown(socket.SHUT_WR)
        MatchServiceRequestHandler(service, ("127.0.0.1", 0), server)
        service.close()
        chunks: list[bytes] = []
        while True:
            chunk = client.recv(65536)
            if not chunk:
                return b"".join(chunks)
            chunks.append(chunk)
    finally:
        client.close()


def test_the_handler_serves_over_a_real_socket() -> None:
    """GET and POST both travel the stdlib socket shell."""
    with FakeHost():
        conn = FakeConnection()
        server = MatchServiceServer(("127.0.0.1", 0), conn)
        try:
            body = _submission("wired")
            request = (
                b"POST /batches HTTP/1.0\r\nContent-Length: "
                + str(len(body)).encode()
                + b"\r\n\r\n"
                + body
            )
            response = _drive_handler(server, request)
            assert b"201" in response.split(b"\r\n", 1)[0]
            response = _drive_handler(server, b"GET /batches/wired HTTP/1.0\r\n\r\n")
            assert b'"queued": 2' in response
        finally:
            server.server_close()


def test_the_entry_point_binds_serves_and_closes() -> None:
    """main opens the connection, announces the port, and shuts down clean."""
    with FakeHost() as host:
        conn = FakeConnection()
        served: list[BaseServer] = []

        def fake_serve(server: BaseServer) -> None:
            served.append(server)

        saved_connect = service_hooks.connect
        saved_serve = host_hooks.serve_forever
        service_hooks.connect = lambda dsn: conn
        host_hooks.serve_forever = fake_serve
        try:
            assert main(["dsn://q", "0"]) == 0
        finally:
            service_hooks.connect = saved_connect
            host_hooks.serve_forever = saved_serve
        assert len(served) == 1
        assert any("listening on http://127.0.0.1" in line for line in host.printed)
        assert conn.closed is True


def test_a_bad_argument_count_prints_usage() -> None:
    with FakeHost() as host:
        assert main(["a", "b", "c"]) == 2
        assert any(line.startswith("usage: match_service") for line in host.printed)
        assert SERVICE_PORT_DEFAULT == 27501


def test_a_sandbox_batch_submits_with_no_map() -> None:
    """An empty map path queues the engine's own sandbox, matchless."""
    conn = FakeConnection()
    body = render_json(
        {
            "name": "sandboxed",
            "jobs": _JOBS_TEXT,
            "lockstep": 75,
            "map_path": "",
            "difficulty": 0,
            "pin_delta": 0,
            "fast_forward": 0,
        }
    ).encode("utf-8")
    status, _kind, payload = route_service_request(conn, "POST", "/batches", body)
    assert status == 201
    assert parse_object(payload.decode("utf-8"))["queued"] == 2


def test_a_handler_bound_to_a_foreign_server_stops_loudly() -> None:
    """The wiring-bug guard: not a request fault, a construction fault."""
    plain = HTTPServer(("127.0.0.1", 0), MatchServiceRequestHandler)
    try:
        client, service = socket.socketpair()
        try:
            client.sendall(b"GET /healthz HTTP/1.0\r\n\r\n")
            client.shutdown(socket.SHUT_WR)
            with pytest.raises(MatchServiceError) as caught:
                MatchServiceRequestHandler(service, ("127.0.0.1", 0), plain)
            assert caught.value.code == "RW-SERVICE-002"
        finally:
            service.close()
            client.close()
    finally:
        plain.server_close()


def test_the_module_guard_runs_main() -> None:
    """The entry point is runnable as a module, like every script here."""
    with FakeHost() as host:
        argv = sys.argv
        already_imported = sys.modules.pop("scripts.match_service")
        sys.argv = ["match_service"]
        try:
            with pytest.raises(SystemExit) as stop:
                runpy.run_module("scripts.match_service", run_name="__main__")
            assert stop.value.code == 2
        finally:
            sys.argv = argv
            sys.modules["scripts.match_service"] = already_imported
        assert len(host.printed) == 1
