"""The TV seam: one SOAP request to a DLNA media renderer.

The fleet sends its four-up stream to the operator's TV by speaking
UPnP AVTransport to it (:mod:`tankpit_bot.service.tv`). Every request
is one HTTP POST of an XML envelope and one XML answer, so that POST is
the whole of what reaches the network, and it is the seam: production
posts to the TV, and tests post to a real HTTP server they run, which
answers as the TV does, refuses as it does, and stalls as a TV that has
gone to sleep does.
"""

from __future__ import annotations

import http.client
import urllib.parse
from typing import Protocol

from platform_core.json_utils import JSONObject, require_int, require_str
from typing_extensions import TypedDict


class SoapReplyDict(TypedDict):
    """A renderer's answer to one SOAP request.

    Attributes:
        status: The HTTP status code. A UPnP fault arrives as 500 with
            a fault envelope, so a non-200 still has a body to read.
        body: The response body as text.
    """

    status: int
    body: str


def encode_soap_reply(reply: SoapReplyDict) -> JSONObject:
    """Encode :class:`SoapReplyDict` to a JSON-serializable dict.

    Args:
        reply: The reply.

    Returns:
        JSON object carrying both fields.
    """
    return {"status": reply["status"], "body": reply["body"]}


def decode_soap_reply(data: JSONObject) -> SoapReplyDict:
    """Validate and decode a :class:`SoapReplyDict` from JSON.

    Args:
        data: JSON object to validate.

    Returns:
        The reply.

    Raises:
        JSONTypeError: A field is missing or has the wrong type.
    """
    return SoapReplyDict(status=require_int(data, "status"), body=require_str(data, "body"))


class PostSoapProtocol(Protocol):
    """Posts one SOAP envelope to a renderer's control URL."""

    def __call__(self, url: str, soap_action: str, body: bytes, timeout: float) -> SoapReplyDict:
        """Send the envelope and read the whole answer.

        Args:
            url: The service's ``http://`` control URL.
            soap_action: The ``SOAPACTION`` header's value, quotes
                included.
            body: The UTF-8 XML envelope.
            timeout: Seconds to wait for the connection and for each
                read.

        Returns:
            The status and body, whatever the status.

        Raises:
            ValueError: The URL is not an ``http://`` URL with a host.
            OSError: The renderer could not be reached, or did not
                answer within ``timeout`` (``TimeoutError``).
        """
        ...


def _real_post_soap(url: str, soap_action: str, body: bytes, timeout: float) -> SoapReplyDict:
    """Post over a plain HTTP connection, as every DLNA renderer serves.

    ``http.client`` rather than ``urllib.request`` because a UPnP fault
    is a 500 WITH a body worth reading, and ``urllib`` turns every
    non-2xx into an exception; this returns the status and the body
    and lets the caller read the fault.

    Args:
        url: The service's ``http://`` control URL.
        soap_action: The ``SOAPACTION`` header's value, quotes included.
        body: The UTF-8 XML envelope.
        timeout: Seconds to wait for the connection and for each read.

    Returns:
        The status and body.

    Raises:
        ValueError: The URL is not an ``http://`` URL with a host.
        OSError: The renderer could not be reached or timed out.
    """
    parts = urllib.parse.urlsplit(url)
    if parts.scheme != "http" or not parts.hostname:
        raise ValueError(f"a renderer control URL is http://host[:port]/path, got {url!r}")
    connection = http.client.HTTPConnection(parts.hostname, parts.port or 80, timeout=timeout)
    try:
        connection.request(
            "POST",
            parts.path or "/",
            body=body,
            headers={
                "Content-Type": 'text/xml; charset="utf-8"',
                "SOAPACTION": soap_action,
            },
        )
        response = connection.getresponse()
        return SoapReplyDict(
            status=response.status,
            body=response.read().decode("utf-8", errors="replace"),
        )
    finally:
        connection.close()


post_soap: PostSoapProtocol = _real_post_soap


__all__ = [
    "PostSoapProtocol",
    "SoapReplyDict",
    "_real_post_soap",
    "decode_soap_reply",
    "encode_soap_reply",
    "post_soap",
]
