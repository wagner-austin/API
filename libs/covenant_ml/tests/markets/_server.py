"""Real socket server serving serialized native API responses in integration tests."""

from __future__ import annotations

from collections.abc import Callable, Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from socket import SHUT_RDWR, socket
from threading import Thread
from urllib.parse import urlsplit

import pytest
from platform_core.json_utils import JSONObject, dump_json_str, load_json_str, narrow_json_to_dict


class ApiServer(HTTPServer):
    """Programmable HTTP origin; production HTTP and JSON code remain in use."""

    documents: dict[str, list[JSONObject]]
    requests: list[str]
    request_headers: list[dict[str, str]]
    statuses: dict[str, int]
    bodies: list[JSONObject]
    disconnect_paths: set[str]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), ApiHandler)
        self.documents = {}
        self.requests = []
        self.request_headers = []
        self.statuses = {}
        self.bodies = []
        self.disconnect_paths = set()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class ApiHandler(BaseHTTPRequestHandler):
    server: ApiServer
    connection: socket

    def __getattr__(self, name: str) -> Callable[[], None]:
        if name in ("do_GET", "do_POST"):
            return self.respond
        raise AttributeError(name)

    def respond(self) -> None:
        self.server.requests.append(self.path)
        self.server.request_headers.append(dict(self.headers.items()))
        path = urlsplit(self.path).path
        if self.command == "POST":
            length = int(self.headers["Content-Length"])
            self.server.bodies.append(
                narrow_json_to_dict(load_json_str(self.rfile.read(length).decode()))
            )
        if path in self.server.disconnect_paths:
            self.connection.shutdown(SHUT_RDWR)
            return
        if path not in self.server.documents:
            self.send_response(404)
            self.end_headers()
            return
        document = self.server.documents[path].pop(0)
        payload = dump_json_str(document).encode("utf-8")
        self.send_response(self.server.statuses.get(path, 200))
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


@pytest.fixture()
def api_server() -> Generator[ApiServer, None, None]:
    server = ApiServer()
    thread = Thread(target=server.serve_forever, args=(0.01,), daemon=True)
    thread.start()
    yield server
    server.shutdown()
    server.server_close()
    thread.join(timeout=5)


__all__ = ["ApiServer", "api_server"]
