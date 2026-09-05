"""Answering HTTP requests for one stream's HLS files.

The encoder (:mod:`tankpit_bot.stream.capture`) writes a playlist and
a rolling window of segments into a directory; this module turns "give
me ``<filename>`` from ``<that directory>``" into a complete HTTP
answer. It is shared by the two surfaces that serve video — the bot
service's own ``/video/{file}`` and the fleet manager's
``/demo/video/{slot}/{file}`` — precisely so the two cannot drift on
status semantics, content types or cache policy.

The filename never touches the filesystem until it has matched one of
exactly two shapes: the playlist's fixed name, or the segment
template's ``seg<5 digits>.ts``. Everything else — including anything
with a separator in it — is a 404 before any path is built, which is
the whole traversal story.
"""

from __future__ import annotations

import re
from pathlib import Path

from aiohttp import web
from platform_core.logging import get_logger
from typing_extensions import TypedDict

from tankpit_bot import _test_hooks as root_hooks
from tankpit_bot.stream.capture import HLS_PLAYLIST_FILENAME

log = get_logger(__name__)

SEGMENT_NAME_PATTERN = re.compile(r"seg\d{5}\.ts")
"""What a segment filename looks like. Must agree with
:data:`tankpit_bot.stream.capture.HLS_SEGMENT_TEMPLATE` — the encoder
names them, this admits them."""

PLAYLIST_CONTENT_TYPE = "application/vnd.apple.mpegurl"
"""The m3u8 media type players key their parsing on."""

SEGMENT_CONTENT_TYPE = "video/mp2t"
"""MPEG transport stream, the container inside each segment."""

PLAYLIST_CACHE_CONTROL = "no-store"
"""The playlist is the live edge; a cached copy is a frozen picture."""

SEGMENT_CACHE_CONTROL = "public, max-age=60, immutable"
"""A segment's name is never reused within a session and its bytes
never change once the encoder's atomic rename lands, so every cache
between here and the viewer may keep it for the minute it stays
relevant."""

WARMUP_RETRY_SECONDS = 3
"""Retry-After on the 503 served before the encoder's first playlist.

Same number, same meaning as the old relay's child-warmup answer: a
bot is registered the moment it forks, its browser takes seconds to
come up, and the first playlist lands a segment-length after that.
"Asked too early" is the ordinary case for a viewer who just pressed
spawn, not a fault."""


class HlsAnswerDict(TypedDict):
    """One complete HTTP answer for an HLS file request.

    An in-process bundle (like
    :class:`~tankpit_bot.runtime_artifacts.BotRunArtifactsDict`), not a
    wire shape — it exists so the pure read logic is testable without
    aiohttp and the aiohttp adapter is one honest translation.

    Attributes:
        status: HTTP status code.
        content_type: Media type of ``body``.
        body: Response bytes.
        cache_control: ``Cache-Control`` header value.
        retry_after_seconds: ``Retry-After`` header value; ``0`` means
            the header is not sent.
    """

    status: int
    content_type: str
    body: bytes
    cache_control: str
    retry_after_seconds: int


def read_hls_file(hls_dir: Path, filename: str) -> HlsAnswerDict:
    """Read one HLS file into a complete HTTP answer.

    The ``FileNotFoundError`` arms are typed translations, not
    swallows: an absent playlist means the encoder has not produced
    one yet (503, come back), an absent segment means the live window
    rotated past it (404, ask the playlist again), and both races are
    the protocol working. Every other failure propagates.

    Args:
        hls_dir: Directory the encoder writes into.
        filename: Requested filename, straight from the URL.

    Returns:
        The answer to serve.
    """
    if filename == HLS_PLAYLIST_FILENAME:
        try:
            body = root_hooks.read_bytes_from(hls_dir / filename, 0)
        except FileNotFoundError:
            log.info("HLS: no playlist yet in %s", hls_dir)
            return HlsAnswerDict(
                status=503,
                content_type="text/plain",
                body=b"stream is starting; no playlist yet",
                cache_control=PLAYLIST_CACHE_CONTROL,
                retry_after_seconds=WARMUP_RETRY_SECONDS,
            )
        return HlsAnswerDict(
            status=200,
            content_type=PLAYLIST_CONTENT_TYPE,
            body=body,
            cache_control=PLAYLIST_CACHE_CONTROL,
            retry_after_seconds=0,
        )
    if SEGMENT_NAME_PATTERN.fullmatch(filename) is None:
        log.info("HLS: refused filename %r", filename)
        return HlsAnswerDict(
            status=404,
            content_type="text/plain",
            body=b"no such stream file",
            cache_control=PLAYLIST_CACHE_CONTROL,
            retry_after_seconds=0,
        )
    try:
        body = root_hooks.read_bytes_from(hls_dir / filename, 0)
    except FileNotFoundError:
        log.info("HLS: segment %r rotated out of %s", filename, hls_dir)
        return HlsAnswerDict(
            status=404,
            content_type="text/plain",
            body=b"segment no longer in the live window",
            cache_control=PLAYLIST_CACHE_CONTROL,
            retry_after_seconds=0,
        )
    return HlsAnswerDict(
        status=200,
        content_type=SEGMENT_CONTENT_TYPE,
        body=body,
        cache_control=SEGMENT_CACHE_CONTROL,
        retry_after_seconds=0,
    )


def hls_web_response(answer: HlsAnswerDict) -> web.Response:
    """Translate an :class:`HlsAnswerDict` into an aiohttp response.

    Args:
        answer: The answer to serve.

    Returns:
        The response, headers and all.
    """
    headers = {"Cache-Control": answer["cache_control"]}
    if answer["retry_after_seconds"]:
        headers["Retry-After"] = str(answer["retry_after_seconds"])
    return web.Response(
        status=answer["status"],
        body=answer["body"],
        content_type=answer["content_type"],
        headers=headers,
    )


__all__ = [
    "PLAYLIST_CACHE_CONTROL",
    "PLAYLIST_CONTENT_TYPE",
    "SEGMENT_CACHE_CONTROL",
    "SEGMENT_CONTENT_TYPE",
    "SEGMENT_NAME_PATTERN",
    "WARMUP_RETRY_SECONDS",
    "HlsAnswerDict",
    "hls_web_response",
    "read_hls_file",
]
