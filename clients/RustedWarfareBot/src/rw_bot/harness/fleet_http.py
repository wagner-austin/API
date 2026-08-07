"""HTTP surface for the match fleet — stdlib only, single-threaded.

This package deliberately carries no web framework, so the fleet serves
with :mod:`http.server`. The server is single-threaded on purpose: every
operation is a dict lookup, a process poll, or a file read, all
sub-millisecond, and the package imports no threading anywhere (see the
coverage note in ``pyproject.toml``).

All decisions live in :func:`route_fleet_request`, a pure function from
``(method, path, body)`` to ``(status, content type, payload)`` — the
socket handler is a thin shell around it. Request bodies are parsed with
:func:`rw_bot.wire.ndjson.parse_object`, the package's typed owner of
flat JSON objects. Responses are rendered by :func:`render_json`, its
emit-side sibling: the same discipline (flat values, no nesting beyond
the two list shapes the fleet actually serves) keeps the whole surface
typed end to end without ``json``'s ``Any``.

Routes, shared verbatim by the control page and the operating AI:

* ``GET  /`` — the control page (:mod:`rw_bot.harness.fleet_page`).
* ``GET  /bots`` — every match's row.
* ``POST /bots`` — spawn: ``{"instance": "a", "seed": 0, "map": "",
  "opponents": 1, "difficulty": 0, "fastforward": 0, "tree": ""}``
  (all but ``instance`` optional).
* ``GET  /bots/{instance}/stats`` — transcript tail + verdict.
* ``POST /bots/{instance}/stop`` — kill the match's process tree.
* ``POST /bots/{instance}/restart`` — respawn a finished match.
* ``DELETE /bots/{instance}`` — drop a finished match.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from http.server import BaseHTTPRequestHandler, HTTPServer

from rw_bot.harness import _test_hooks
from rw_bot.harness.fleet import FleetError, FleetManager, FleetMatchRow, FleetStats
from rw_bot.harness.fleet_page import FLEET_PAGE_HTML
from rw_bot.wire.ndjson import NdjsonError, parse_object

FLEET_PORT_DEFAULT = 27500

_JSON = "application/json"
_HTML = "text/html; charset=utf-8"
_TEXT = "text/plain; charset=utf-8"

_BAD_BODY = "RW-FLEET-007"
_BAD_FIELD = "RW-FLEET-008"

#: Error codes the HTTP layer maps to 409 (a state conflict the caller
#: can resolve); bad spawn input is 400; everything else is a 404.
_CONFLICT_CODES = ("RW-FLEET-003", "RW-FLEET-005", "RW-FLEET-006")
_INPUT_CODES = ("RW-FLEET-001", "RW-FLEET-002", _BAD_BODY, _BAD_FIELD)

_STRING_ESCAPES = {
    "\\": "\\\\",
    '"': '\\"',
    "\b": "\\b",
    "\f": "\\f",
    "\n": "\\n",
    "\r": "\\r",
    "\t": "\\t",
}


def _render_string(value: str) -> str:
    """Render one JSON string literal.

    Args:
        value: The string to render.

    Returns:
        The quoted, escaped literal.
    """
    rendered: list[str] = ['"']
    for character in value:
        if character in _STRING_ESCAPES:
            rendered.append(_STRING_ESCAPES[character])
        elif ord(character) < 0x20:
            rendered.append(f"\\u{ord(character):04x}")
        else:
            rendered.append(character)
    rendered.append('"')
    return "".join(rendered)


def render_json(
    payload: Mapping[
        str,
        str | int | bool | None | Sequence[str] | Sequence[Mapping[str, str | int | bool | None]],
    ],
) -> str:
    """Render one response object as JSON.

    The emit-side sibling of :func:`rw_bot.wire.ndjson.parse_object`:
    the value grammar is exactly what the fleet's responses carry —
    flat scalars, a list of strings (the report lines), or a list of
    flat objects (the match rows). Anything else is a programming
    error, not data.

    Args:
        payload: The response object.

    Returns:
        Its JSON text.
    """
    parts: list[str] = []
    for key, value in payload.items():
        parts.append(f"{_render_string(key)}: {_render_value(value)}")
    return "{" + ", ".join(parts) + "}"


def _render_value(
    value: str
    | int
    | bool
    | None
    | Sequence[str]
    | Sequence[Mapping[str, str | int | bool | None]],
) -> str:
    """Render one value of the shapes the fleet serves.

    Args:
        value: A flat scalar, a list of strings, or a list of flat
            objects.

    Returns:
        Its JSON rendering.
    """
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, str):
        return _render_string(value)
    items: list[str] = []
    for item in value:
        items.append(_render_string(item) if isinstance(item, str) else render_json(item))
    return "[" + ", ".join(items) + "]"


def _encode_row(row: FleetMatchRow) -> dict[str, str | int | bool | None]:
    """Flatten one match row for :func:`render_json`.

    Args:
        row: The typed row.

    Returns:
        The same fields as a plain flat mapping.
    """
    return {
        "instance": row["instance"],
        "seed": row["seed"],
        "map": row["map"],
        "opponents": row["opponents"],
        "difficulty": row["difficulty"],
        "fastforward": row["fastforward"],
        "tree": row["tree"],
        "pid": row["pid"],
        "alive": row["alive"],
        "returncode": row["returncode"],
    }


def _encode_stats(stats: FleetStats) -> dict[str, bool | str | list[str]]:
    """Flatten one stats summary for :func:`render_json`.

    Args:
        stats: The typed summary.

    Returns:
        The same fields as a plain mapping.
    """
    return {
        "available": stats["available"],
        "finished": stats["finished"],
        "verdict": stats["verdict"],
        "report": stats["report"],
    }


def _spawn_fields(body: bytes) -> dict[str, int | str]:
    """Parse and validate a spawn request body.

    Args:
        body: Raw request body — one flat JSON object.

    Returns:
        Keyword arguments for :meth:`FleetManager.spawn`.

    Raises:
        FleetError: When the body is not the flat object grammar, a
            field has the wrong type, or a required field is missing.
    """
    try:
        fields = parse_object(body.decode("utf-8"))
    except (UnicodeDecodeError, NdjsonError) as error:
        raise FleetError(_BAD_BODY, f"unreadable spawn body: {error}") from error
    instance = fields.get("instance", "")
    if not isinstance(instance, str) or not instance:
        raise FleetError(_BAD_FIELD, "spawn body must carry a non-empty 'instance'")
    out: dict[str, int | str] = {"instance": instance}
    for name, target in (("map", "map_name"), ("tree", "tree")):
        value = fields.get(name, "")
        if not isinstance(value, str):
            raise FleetError(_BAD_FIELD, f"spawn field {name!r} must be a string")
        out[target] = value
    for name in ("seed", "opponents", "difficulty", "fastforward"):
        default = 1 if name == "opponents" else 0
        value = fields.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int):
            raise FleetError(_BAD_FIELD, f"spawn field {name!r} must be an integer")
        out[name] = value
    return out


def route_fleet_request(
    manager: FleetManager, method: str, path: str, body: bytes
) -> tuple[int, str, bytes]:
    """Decide one request. Pure: no sockets, no globals.

    Every :class:`FleetError` any operation raises becomes a 4xx with
    the error's own text as the body, announced through the output
    line — one conversion point for the whole surface.

    Args:
        manager: The fleet registry to operate on.
        method: HTTP method, upper-case.
        path: Request path.
        body: Raw request body (empty for bodiless requests).

    Returns:
        ``(status, content type, payload)``.
    """
    try:
        return _decide(manager, method, path, body)
    except FleetError as error:
        if error.code in _CONFLICT_CODES:
            status = 409
        elif error.code in _INPUT_CODES:
            status = 400
        else:
            status = 404
        _test_hooks.write_line(f"[fleet] refused {method} {path} ({status}): {error}")
        return status, _TEXT, str(error).encode("utf-8")


def _decide(manager: FleetManager, method: str, path: str, body: bytes) -> tuple[int, str, bytes]:
    """Route one request, letting FleetError propagate.

    Args:
        manager: The fleet registry to operate on.
        method: HTTP method, upper-case.
        path: Request path.
        body: Raw request body.

    Returns:
        ``(status, content type, payload)``.
    """
    if method == "GET" and path == "/":
        return 200, _HTML, FLEET_PAGE_HTML.encode("utf-8")
    if method == "GET" and path == "/bots":
        rows = [_encode_row(row) for row in manager.report()]
        return 200, _JSON, render_json({"bots": rows}).encode("utf-8")
    if method == "POST" and path == "/bots":
        fields = _spawn_fields(body)
        row = manager.spawn(
            instance=str(fields["instance"]),
            seed=int(fields["seed"]),
            map_name=str(fields["map_name"]),
            opponents=int(fields["opponents"]),
            difficulty=int(fields["difficulty"]),
            fastforward=int(fields["fastforward"]),
            tree=str(fields["tree"]),
        )
        return 201, _JSON, render_json(_encode_row(row)).encode("utf-8")
    parts = path.strip("/").split("/")
    if len(parts) == 3 and parts[0] == "bots" and method == "GET" and parts[2] == "stats":
        stats = manager.stats(parts[1])
        return 200, _JSON, render_json(_encode_stats(stats)).encode("utf-8")
    if len(parts) == 3 and parts[0] == "bots" and method == "POST" and parts[2] == "stop":
        return 200, _JSON, render_json(_encode_row(manager.stop(parts[1]))).encode("utf-8")
    if len(parts) == 3 and parts[0] == "bots" and method == "POST" and parts[2] == "restart":
        return 201, _JSON, render_json(_encode_row(manager.restart(parts[1]))).encode("utf-8")
    if len(parts) == 2 and parts[0] == "bots" and method == "DELETE":
        return 200, _JSON, render_json(_encode_row(manager.remove(parts[1]))).encode("utf-8")
    return 404, _TEXT, f"no route for {method} {path}".encode()


class FleetServer(HTTPServer):
    """The fleet's HTTP server, carrying its manager to the handlers."""

    def __init__(self, address: tuple[str, int], manager: FleetManager) -> None:
        """Bind the server.

        Args:
            address: ``(host, port)`` to bind.
            manager: The fleet registry every request operates on.
        """
        super().__init__(address, FleetRequestHandler)
        self.fleet_manager = manager


class FleetRequestHandler(BaseHTTPRequestHandler):
    """Thin socket shell around :func:`route_fleet_request`."""

    def _dispatch(self, method: str) -> None:
        """Read one request, route it, write the response.

        Args:
            method: HTTP method, upper-case.

        Raises:
            FleetError: If the handler is serving for something other
                than a :class:`FleetServer` — a wiring bug, not a
                request fault.
        """
        server = self.server
        if not isinstance(server, FleetServer):
            raise FleetError("RW-FLEET-009", "handler bound to a non-fleet server")
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length) if length else b""
        status, content_type, payload = route_fleet_request(
            server.fleet_manager, method, self.path, body
        )
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        """Serve one GET."""
        self._dispatch("GET")

    def do_POST(self) -> None:
        """Serve one POST."""
        self._dispatch("POST")

    def do_DELETE(self) -> None:
        """Serve one DELETE."""
        self._dispatch("DELETE")

    def log_message(self, format: str, *args: str | int) -> None:
        """Route the per-request access line through the output hook.

        Args:
            format: printf-style format string.
            *args: Format arguments.
        """
        _test_hooks.write_line(f"[fleet] {self.address_string()} {format % args}")


def resolve_port(args: Sequence[str]) -> int:
    """Resolve the listen port from the command line.

    Args:
        args: Arguments after the program name.

    Returns:
        The ``--port N`` value when given, else
        :data:`FLEET_PORT_DEFAULT`.
    """
    if len(args) == 2 and args[0] == "--port":
        return int(args[1])
    return FLEET_PORT_DEFAULT


def main() -> int:
    """Run the ``rw-fleet`` manager until interrupted.

    ``--port N`` overrides :data:`FLEET_PORT_DEFAULT`. Binds loopback
    only — this is a desktop control surface, not a network service.

    Returns:
        Process exit status.
    """
    port = resolve_port(_test_hooks.read_argv())
    server = FleetServer(("127.0.0.1", port), FleetManager())
    _test_hooks.write_line(f"[fleet] rw-fleet listening on http://127.0.0.1:{port}/")
    _test_hooks.serve_forever(server)
    server.server_close()
    return 0


__all__ = [
    "FLEET_PORT_DEFAULT",
    "FleetRequestHandler",
    "FleetServer",
    "main",
    "render_json",
    "resolve_port",
    "route_fleet_request",
]
