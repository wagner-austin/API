"""Real socket server serving serialized native API responses in integration tests."""

from __future__ import annotations

from collections.abc import Callable, Generator
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from urllib.parse import urlsplit

import pytest
from platform_core.json_utils import JSONObject, dump_json_str


class ApiServer(HTTPServer):
    """Programmable HTTP origin; production HTTP and JSON code remain in use."""

    documents: dict[str, list[JSONObject]]
    requests: list[str]
    request_headers: list[dict[str, str]]
    statuses: dict[str, int]

    def __init__(self) -> None:
        super().__init__(("127.0.0.1", 0), ApiHandler)
        self.documents = {}
        self.requests = []
        self.request_headers = []
        self.statuses = {}

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_port}"


class ApiHandler(BaseHTTPRequestHandler):
    server: ApiServer

    def __getattr__(self, name: str) -> Callable[[], None]:
        if name == "do_GET":
            return self.respond
        raise AttributeError(name)

    def respond(self) -> None:
        self.server.requests.append(self.path)
        self.server.request_headers.append(dict(self.headers.items()))
        path = urlsplit(self.path).path
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
