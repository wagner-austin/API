"""A real HTTP server that answers AVTransport requests as the TV does.

The TV seam (:mod:`tankpit_bot.service._test_hooks.tv`) is one POST
over a real socket, so its tests run it against a real server on the
loopback interface rather than a stand-in for the call. The answers are
the shapes the operator's Samsung UN50NU6900 gave on 2026-10-17: an
empty response envelope per action, and the transport state inside
``GetTransportInfoResponse``. A server can also fault an action the way
a UPnP renderer does (HTTP 500 with an ``errorCode``), or stall without
answering, the way a TV in standby holds a connection open.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socketserver import BaseServer
from types import TracebackType

from typing_extensions import TypedDict

from tankpit_bot.service.tv import AV_TRANSPORT


class SoapRequestDict(TypedDict):
    """One request the server received.

    Attributes:
        path: The request path.
        soap_action: The ``SOAPACTION`` header.
        content_type: The ``Content-Type`` header.
        body: The request body as text.
    """

    path: str
    soap_action: str
    content_type: str
    body: str


def response_envelope(action: str, inner: str) -> str:
    """Build the envelope a renderer answers an action with.

    Args:
        action: The action answered.
        inner: The response's argument elements.

    Returns:
        The envelope.
    """
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"><s:Body>'
        f'<u:{action}Response xmlns:u="{AV_TRANSPORT}">{inner}</u:{action}Response>'
        "</s:Body></s:Envelope>"
    )


def fault_envelope(code: int, description: str) -> str:
    """Build the UPnP fault a renderer answers a refused action with.

    Args:
        code: The UPnP error code.
        description: Its description.

    Returns:
        The fault envelope.
    """
    return (
        '<?xml version="1.0"?><s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/">'
        "<s:Body><s:Fault><faultcode>s:Client</faultcode><faultstring>UPnPError</faultstring>"
        '<detail><UPnPError xmlns="urn:schemas-upnp-org:control-1-0">'
        f"<errorCode>{code}</errorCode><errorDescription>{description}</errorDescription>"
        "</UPnPError></detail></s:Fault></s:Body></s:Envelope>"
    )


class TvServer(ThreadingHTTPServer):
    """A loopback renderer: answers by action, records every request.

    Attributes:
        answers: Status and body per action name; an action not listed
            is answered with an empty 200 response envelope.
        stall: When set, every request is held unanswered until the
            server closes.
        requests: Every request received, in order.
    """

    daemon_threads = True

    def __init__(self) -> None:
        """Bind to a free loopback port; serve nothing yet."""
        super().__init__(("127.0.0.1", 0), _Handler)
        self.answers: dict[str, tuple[int, str]] = {}
        self.stall = False
        self.requests: list[SoapRequestDict] = []
        self.released = threading.Event()
        self._thread = threading.Thread(target=self.serve_forever, daemon=True)

    @property
    def control_url(self) -> str:
        """The URL a client posts AVTransport requests to."""
        return f"http://127.0.0.1:{self.server_address[1]}/upnp/control/AVTransport1"

    def answer_state(self, state: str) -> None:
        """Make ``GetTransportInfo`` report ``state``.

        Args:
            state: The ``CurrentTransportState`` to report.
        """
        self.answers["GetTransportInfo"] = (
            200,
            response_envelope(
                "GetTransportInfo",
                f"<CurrentTransportState>{state}</CurrentTransportState>"
                "<CurrentTransportStatus>OK</CurrentTransportStatus><CurrentSpeed>1</CurrentSpeed>",
            ),
        )

    def __enter__(self) -> TvServer:
        """Start serving on a background thread.

        Returns:
            This server.
        """
        self._thread.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Release any stalled request and stop serving.

        Args:
            exc_type: Ignored.
            exc: Ignored.
            traceback: Ignored.
        """
        self.released.set()
        self.shutdown()
        self.server_close()
        self._thread.join()


def _tv_server(server: BaseServer) -> TvServer:
    """Narrow a handler's server to the renderer it belongs to.

    Args:
        server: The handler's server.

    Returns:
        The same server, typed.

    Raises:
        TypeError: The handler is mounted on some other server.
    """
    if not isinstance(server, TvServer):
        raise TypeError(f"the AVTransport handler runs on a TvServer, not {type(server)!r}")
    return server


class _Handler(BaseHTTPRequestHandler):
    """Answers one POST from the server's script."""

    def do_POST(self) -> None:
        """Record the request and answer it, or hold it if the server stalls."""
        server = _tv_server(self.server)
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length).decode("utf-8")
        soap_action = self.headers.get("SOAPACTION", "")
        server.requests.append(
            SoapRequestDict(
                path=self.path,
                soap_action=soap_action,
                content_type=self.headers.get("Content-Type", ""),
                body=body,
            )
        )
        if server.stall:
            server.released.wait()
            return
        action = soap_action.strip('"').rsplit("#", 1)[-1]
        status, reply = server.answers.get(action, (200, response_envelope(action, "")))
        payload = reply.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", 'text/xml; charset="utf-8"')
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)


__all__ = [
    "SoapRequestDict",
    "TvServer",
    "fault_envelope",
    "response_envelope",
]
