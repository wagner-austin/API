"""Sending the fleet's four-up stream to the operator's TV.

The TV is a DLNA media renderer: it plays whatever URL it is told to,
over UPnP AVTransport. Three requests do the whole job, and they are
the ones a phone's "cast" button sends: ``SetAVTransportURI`` names
the stream, ``Play`` starts it, and ``GetTransportInfo`` reads back
whether the TV is playing. ``Stop`` ends it. Measured against the
operator's Samsung UN50NU6900 on 2026-10-17: the stream's HLS playlist
URL, offered as ``application/vnd.apple.mpegurl``, played with no
prompt on the TV.

Both addresses come from the fleet's environment and nothing else.
``TANKPIT_TV_CONTROL_URL`` is the renderer's AVTransport control URL
on the home network; ``TANKPIT_MOSAIC_URL`` is where the TV fetches
the stream, which is the public edge's ``/demo/mosaic/`` because the
TV can reach the internet and cannot reach the container. Unset means
there is no TV, and a cast says so by name rather than guessing one.

Every failure carries a code: ``TV_NOT_CONFIGURED`` (no addresses),
``TV_REFUSED`` (the renderer answered with a fault), and
``TV_UNREADABLE`` (it answered without a transport state). An answer
that is not XML at all raises ``ElementTree.ParseError``, and a
renderer that cannot be reached raises ``OSError`` from the seam
itself; both propagate unchanged to the route boundary
(:mod:`tankpit_bot.service.tv_routes`), which names them.
"""

from __future__ import annotations

from xml.etree import ElementTree
from xml.sax.saxutils import escape

from platform_core.json_utils import JSONObject, require_str
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks as top_hooks
from tankpit_bot.service import _test_hooks as service_hooks
from tankpit_bot.stream.hls import PLAYLIST_CONTENT_TYPE

AV_TRANSPORT = "urn:schemas-upnp-org:service:AVTransport:1"
"""The UPnP service every request here is addressed to."""

TV_TIMEOUT_SECONDS = 10.0
"""How long one request may take. A TV answers in well under a second
when it is on; one in standby does not answer at all, and the operator
should hear that in seconds, not wait on it."""

MEDIA_TITLE = "TankPit live"
"""The title the TV shows for the stream."""


class TvError(Exception):
    """A TV request that failed for a reason this module can name.

    Attributes:
        code: ``TV_NOT_CONFIGURED``, ``TV_REFUSED`` or ``TV_UNREADABLE``.
    """

    def __init__(self, code: str, message: str) -> None:
        """Record the code beside the message.

        Args:
            code: The failure's code.
            message: What happened, for a person.
        """
        super().__init__(f"{code}: {message}")
        self.code = code


class TvTargetDict(TypedDict):
    """Where the TV is, and where it fetches the stream.

    Attributes:
        control_url: The renderer's AVTransport control URL.
        media_url: The stream's playlist URL as the TV reaches it.
    """

    control_url: str
    media_url: str


class TvStateDict(TypedDict):
    """What the TV reported after a request.

    Attributes:
        transport_state: The renderer's ``CurrentTransportState``, as it
            spells it (``PLAYING``, ``STOPPED``, ``TRANSITIONING``, ...).
        media_url: The stream URL the request concerned.
    """

    transport_state: str
    media_url: str


def encode_tv_target(target: TvTargetDict) -> JSONObject:
    """Encode :class:`TvTargetDict` to a JSON-serializable dict.

    Args:
        target: The target.

    Returns:
        JSON object carrying both fields.
    """
    return {"control_url": target["control_url"], "media_url": target["media_url"]}


def decode_tv_target(data: JSONObject) -> TvTargetDict:
    """Validate and decode a :class:`TvTargetDict` from JSON.

    Args:
        data: JSON object to validate.

    Returns:
        The target.

    Raises:
        JSONTypeError: A field is missing or has the wrong type.
    """
    return TvTargetDict(
        control_url=require_str(data, "control_url"),
        media_url=require_str(data, "media_url"),
    )


def encode_tv_state(state: TvStateDict) -> JSONObject:
    """Encode :class:`TvStateDict` to a JSON-serializable dict.

    Args:
        state: What the TV reported.

    Returns:
        JSON object carrying both fields.
    """
    return {"transport_state": state["transport_state"], "media_url": state["media_url"]}


def decode_tv_state(data: JSONObject) -> TvStateDict:
    """Validate and decode a :class:`TvStateDict` from JSON.

    Args:
        data: JSON object to validate.

    Returns:
        The state.

    Raises:
        JSONTypeError: A field is missing or has the wrong type.
    """
    return TvStateDict(
        transport_state=require_str(data, "transport_state"),
        media_url=require_str(data, "media_url"),
    )


def resolve_tv_target() -> TvTargetDict:
    """Read the TV's address and the stream's URL from the environment.

    Returns:
        The target.

    Raises:
        TvError: ``TV_NOT_CONFIGURED`` when either variable is unset or
            empty.
    """
    control_url = top_hooks.get_env("TANKPIT_TV_CONTROL_URL")
    media_url = top_hooks.get_env("TANKPIT_MOSAIC_URL")
    if not control_url or not media_url:
        raise TvError(
            "TV_NOT_CONFIGURED",
            "TANKPIT_TV_CONTROL_URL and TANKPIT_MOSAIC_URL must both be set to cast",
        )
    return TvTargetDict(control_url=control_url, media_url=media_url)


def _envelope(action: str, arguments: str) -> bytes:
    """Wrap one AVTransport action in a SOAP envelope.

    Args:
        action: The action's name.
        arguments: The action's argument elements after ``InstanceID``,
            already escaped.

    Returns:
        The UTF-8 envelope.
    """
    return (
        '<?xml version="1.0" encoding="utf-8"?>'
        '<s:Envelope xmlns:s="http://schemas.xmlsoap.org/soap/envelope/"'
        ' s:encodingStyle="http://schemas.xmlsoap.org/soap/encoding/"><s:Body>'
        f'<u:{action} xmlns:u="{AV_TRANSPORT}"><InstanceID>0</InstanceID>{arguments}'
        f"</u:{action}></s:Body></s:Envelope>"
    ).encode()


def media_metadata(media_url: str) -> str:
    """Describe the stream to the TV as one video item.

    The renderer decides whether it can play a URL from this item's
    ``protocolInfo``; without it a Samsung renderer guesses from the
    URL's extension.

    Args:
        media_url: The stream's playlist URL.

    Returns:
        The DIDL-Lite document, unescaped.
    """
    return (
        '<DIDL-Lite xmlns="urn:schemas-upnp-org:metadata-1-0/DIDL-Lite/"'
        ' xmlns:dc="http://purl.org/dc/elements/1.1/"'
        ' xmlns:upnp="urn:schemas-upnp-org:metadata-1-0/upnp/">'
        '<item id="0" parentID="-1" restricted="1">'
        f"<dc:title>{escape(MEDIA_TITLE)}</dc:title>"
        "<upnp:class>object.item.videoItem</upnp:class>"
        f'<res protocolInfo="http-get:*:{PLAYLIST_CONTENT_TYPE}:*">{escape(media_url)}</res>'
        "</item></DIDL-Lite>"
    )


def _element_text(body: str, name: str) -> str:
    """Return the text of the first element whose local name is ``name``.

    Args:
        body: An XML document.
        name: The element's local name, without its namespace.

    Returns:
        The element's text; empty when it has none or is absent.

    Raises:
        xml.etree.ElementTree.ParseError: The body is not XML; the
            route boundary answers it as ``TV_UNREADABLE``.
    """
    root = ElementTree.fromstring(body)
    for element in root.iter():
        if element.tag.rsplit("}", 1)[-1] == name:
            return element.text or ""
    return ""


def _call(control_url: str, action: str, arguments: str) -> str:
    """Send one action and return the answer's body.

    Args:
        control_url: The renderer's AVTransport control URL.
        action: The action's name.
        arguments: Its argument elements, already escaped.

    Returns:
        The body of the renderer's 200 answer.

    Raises:
        TvError: ``TV_REFUSED`` when it answers anything but 200, with
            the UPnP error code and description it gave.
        xml.etree.ElementTree.ParseError: A refusal whose body is not
            XML.
        OSError: The renderer could not be reached or timed out.
    """
    reply = service_hooks.post_soap(
        control_url,
        f'"{AV_TRANSPORT}#{action}"',
        _envelope(action, arguments),
        TV_TIMEOUT_SECONDS,
    )
    if reply["status"] != 200:
        code = _element_text(reply["body"], "errorCode")
        description = _element_text(reply["body"], "errorDescription")
        raise TvError(
            "TV_REFUSED",
            f"{action} answered HTTP {reply['status']} (UPnP error {code or 'none'}:"
            f" {description or 'no description'})",
        )
    return reply["body"]


def transport_state(control_url: str) -> str:
    """Ask the TV what it is doing.

    Args:
        control_url: The renderer's AVTransport control URL.

    Returns:
        Its ``CurrentTransportState``.

    Raises:
        TvError: ``TV_REFUSED`` when it faults, ``TV_UNREADABLE`` when
            its answer carries no state.
        xml.etree.ElementTree.ParseError: Its answer is not XML.
        OSError: It could not be reached or timed out.
    """
    state = _element_text(_call(control_url, "GetTransportInfo", ""), "CurrentTransportState")
    if not state:
        raise TvError("TV_UNREADABLE", "GetTransportInfo answered without a CurrentTransportState")
    return state


def cast(target: TvTargetDict) -> TvStateDict:
    """Make the TV play the stream, and report what it says it is doing.

    Args:
        target: Where the TV is and where it fetches the stream.

    Returns:
        The TV's transport state after ``Play``.

    Raises:
        TvError: ``TV_REFUSED`` or ``TV_UNREADABLE``.
        xml.etree.ElementTree.ParseError: An answer is not XML.
        OSError: The TV could not be reached or timed out.
    """
    media_url = target["media_url"]
    _call(
        target["control_url"],
        "SetAVTransportURI",
        f"<CurrentURI>{escape(media_url)}</CurrentURI>"
        f"<CurrentURIMetaData>{escape(media_metadata(media_url))}</CurrentURIMetaData>",
    )
    _call(target["control_url"], "Play", "<Speed>1</Speed>")
    return TvStateDict(transport_state=transport_state(target["control_url"]), media_url=media_url)


def stop(target: TvTargetDict) -> TvStateDict:
    """Make the TV stop playing, and report what it says it is doing.

    Args:
        target: Where the TV is and where it fetches the stream.

    Returns:
        The TV's transport state after ``Stop``.

    Raises:
        TvError: ``TV_REFUSED`` or ``TV_UNREADABLE``.
        xml.etree.ElementTree.ParseError: An answer is not XML.
        OSError: The TV could not be reached or timed out.
    """
    _call(target["control_url"], "Stop", "")
    return TvStateDict(
        transport_state=transport_state(target["control_url"]),
        media_url=target["media_url"],
    )


__all__ = [
    "AV_TRANSPORT",
    "MEDIA_TITLE",
    "TV_TIMEOUT_SECONDS",
    "TvError",
    "TvStateDict",
    "TvTargetDict",
    "cast",
    "decode_tv_state",
    "decode_tv_target",
    "encode_tv_state",
    "encode_tv_target",
    "media_metadata",
    "resolve_tv_target",
    "stop",
    "transport_state",
]
