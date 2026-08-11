"""The match service's HTTP door, in the fleet's own idiom.

One pure router over the queue operations -- no framework, no threads, the
stdlib server, requests parsed by :func:`rw_bot.wire.ndjson.parse_object`
and responses rendered by its emit-side sibling. The surface is the design
page's phase-one control plane ([[harness-match-service]]) in the shape
this repo already ships HTTP: the fleet page proved the pattern and the
guard keeps it typed end to end.

Any session, human or AI, on any machine that can reach the port submits a
batch through one door; workers keep polling the database directly, because
the queue's claim transaction is the coordination point, not this surface.
"""

from __future__ import annotations

from rw_bot.harness import _test_hooks as host_hooks
from rw_bot.harness.sweep import SweepError, parse_jobs
from rw_bot.service._test_hooks import Connection
from rw_bot.service.dashboard import render_dashboard
from rw_bot.service.queue import (
    batch_results,
    batch_status,
    bootstrap,
    reprioritize,
    retry_failed,
    submit,
)
from rw_bot.service.queue_rows import MatchServiceError
from rw_bot.service.submit import batch_config
from rw_bot.validation import DecodeError, require_int, require_non_empty_str, require_str
from rw_bot.wire.ndjson import NdjsonError, parse_object, render_json

_JSON = "application/json"
_NDJSON = "application/x-ndjson"
_TEXT = "text/plain; charset=utf-8"
_HTML = "text/html; charset=utf-8"


def route_service_request(
    conn: Connection, method: str, path: str, body: bytes
) -> tuple[int, str, bytes]:
    """Decide one request. Pure: no sockets, no globals.

    Every decode failure, malformed job line and service error becomes a
    4xx with the
    error's own text as the body -- one conversion point for the whole
    surface, exactly as the fleet router converts its errors.

    Args:
        conn: An open queue connection.
        method: HTTP method, upper-case.
        path: Request path.
        body: Raw request body (empty for bodiless requests).

    Returns:
        ``(status, content type, payload)``.
    """
    try:
        return _decide(conn, method, path, body)
    except (DecodeError, NdjsonError, MatchServiceError, SweepError) as error:
        host_hooks.write_line(f"[service] refused {method} {path} (400): {error}")
        return 400, _TEXT, str(error).encode("utf-8")


def _decide(conn: Connection, method: str, path: str, body: bytes) -> tuple[int, str, bytes]:
    """Route one request, letting the typed errors propagate.

    Args:
        conn: An open queue connection.
        method: HTTP method, upper-case.
        path: Request path.
        body: Raw request body.

    Returns:
        ``(status, content type, payload)``.
    """
    if method == "GET" and path == "/healthz":
        return 200, _JSON, render_json({"ok": True}).encode("utf-8")
    if method == "GET" and path == "/":
        return 200, _HTML, render_dashboard(conn).encode("utf-8")
    if method == "POST" and path == "/batches":
        fields = parse_object(body.decode("utf-8"))
        name = require_non_empty_str(fields, "name")
        jobs = parse_jobs(require_non_empty_str(fields, "jobs").splitlines())
        config = batch_config(
            name,
            require_int(fields, "lockstep"),
            require_str(fields, "map_path"),
            require_int(fields, "difficulty"),
            require_int(fields, "pin_delta"),
            require_int(fields, "fast_forward"),
        )
        bootstrap(conn)
        queued = submit(conn, name, config, jobs)
        payload: dict[str, str | int] = {"batch": name, "queued": queued, "total": len(jobs)}
        return 201, _JSON, render_json(payload).encode("utf-8")
    parts = path.strip("/").split("/")
    if len(parts) == 3 and parts[0] == "batches" and parts[2] == "priority" and method == "POST":
        fields = parse_object(body.decode("utf-8"))
        arm = require_str(fields, "label") if "label" in fields else ""
        moved = reprioritize(conn, parts[1], require_int(fields, "priority"), arm)
        bumped: dict[str, str | int] = {"batch": parts[1], "moved": moved}
        return 200, _JSON, render_json(bumped).encode("utf-8")
    if len(parts) == 3 and parts[0] == "batches" and parts[2] == "retries" and method == "POST":
        requeued = retry_failed(conn, parts[1])
        retried: dict[str, str | int] = {"batch": parts[1], "requeued": requeued}
        return 200, _JSON, render_json(retried).encode("utf-8")
    if len(parts) == 3 and parts[0] == "batches" and parts[2] == "results" and method == "GET":
        lines = tuple(
            render_json(
                {
                    "label": result["label"],
                    "seed": result["seed"],
                    "state": result["state"],
                    "verdict": result["verdict"],
                }
            )
            for result in batch_results(conn, parts[1])
        )
        body_text = "\n".join(lines) + "\n" if lines else ""
        return 200, _NDJSON, body_text.encode("utf-8")
    if len(parts) == 2 and parts[0] == "batches" and method == "GET":
        status = batch_status(conn, parts[1])
        return (
            200,
            _JSON,
            render_json(
                {
                    "batch": status["batch"],
                    "queued": status["queued"],
                    "running": status["running"],
                    "done": status["done"],
                    "failed": status["failed"],
                }
            ).encode("utf-8"),
        )
    return 404, _TEXT, f"no route for {method} {path}".encode()
