from __future__ import annotations

import hashlib
import time
import types
from collections.abc import Callable
from typing import Protocol, runtime_checkable

from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue, load_json_str

from platform_music.models import PlayRecord, Track
from platform_music.services.protocol import MusicServiceProto


@runtime_checkable
class _HttpResp(Protocol):
    def read(self) -> bytes: ...

    def __enter__(self) -> _HttpResp: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> None: ...


class _UrlOpen(Protocol):
    def urlopen(self, req: _RequestObj, timeout: float) -> _HttpResp: ...


class _RequestObj(Protocol):
    def add_header(self, name: str, value: str) -> None: ...


# Hook type for testing
YouTubeHttpPostHook = Callable[[str, str, str, str, float, str], str]


def sapisidhash(sapisid: str, origin: str, now: int) -> str:
    """Generate SAPISIDHASH authentication header value for YouTube API.

    This function is exported for testing the authentication logic.
    """
    msg = f"{now} {sapisid} {origin}".encode()
    sha = hashlib.sha1(msg).hexdigest()
    return f"SAPISIDHASH {now}_{sha}"


def http_post_impl(
    url: str,
    *,
    sapisid: str,
    cookies: str,
    origin: str,
    timeout: float,
    body: str,
) -> str:
    """Production HTTP POST implementation for YouTube Music API.

    Uses urllib.request to make HTTP POST requests with SAPISIDHASH auth.
    This function is exported for use by testing.py to set production hooks.
    """
    imp = __import__
    req_mod = imp("urllib.request", fromlist=["Request"])
    req: _RequestObj = req_mod.Request(url)
    # Attach request body for POST (works for stdlib and our test doubles)
    object.__setattr__(req, "data", body.encode("utf-8"))
    now = int(time.time())
    req.add_header("Authorization", sapisidhash(sapisid, origin, now))
    req.add_header("Cookie", cookies)
    req.add_header("Origin", origin)
    req.add_header("Content-Type", "application/json")
    req.add_header("X-YouTube-Client-Name", "67")  # WEB_REMIX
    req.add_header("X-YouTube-Client-Version", "1.20250101.00.00")
    opener: _UrlOpen = imp("urllib.request", fromlist=["urlopen"])
    with opener.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _http_post(
    url: str,
    *,
    sapisid: str,
    cookies: str,
    origin: str,
    timeout: float,
    body: str,
) -> str:
    """HTTP POST via hook. Hook is always set (production or test fake)."""
    from platform_music.testing import hooks

    return hooks.youtube_http_post(url, sapisid, cookies, origin, timeout, body)


def _decode_yt_item(raw: JSONValue) -> PlayRecord:
    if not isinstance(raw, dict):
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR, message="invalid yt item", http_status=502
        )
    title = raw.get("title")
    artist = raw.get("artist")
    vid = raw.get("videoId")
    played = raw.get("playedAt")
    dur = raw.get("durationSeconds")
    if not (
        isinstance(title, str)
        and isinstance(artist, str)
        and isinstance(played, str)
        and isinstance(dur, int)
    ):
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR, message="invalid yt fields", http_status=502
        )
    track: Track = {
        "id": str(vid) if isinstance(vid, str) else "yt:" + title,
        "title": title,
        "artist_name": artist,
        "duration_ms": int(dur) * 1000,
        "service": "youtube_music",
    }
    return {"track": track, "played_at": played, "service": "youtube_music"}


@runtime_checkable
class YouTubeMusicProto(MusicServiceProto, Protocol):
    pass


class _YouTubeClient(YouTubeMusicProto):
    def __init__(self, *, sapisid: str, cookies: str) -> None:
        self._sid = sapisid
        self._ck = cookies

    def get_listening_history(
        self, *, start_date: str, end_date: str, limit: int | None = None
    ) -> list[PlayRecord]:
        # youtubei history endpoint requires POST with context
        url = "https://music.youtube.com/youtubei/v1/history?prettyPrint=false"
        body = (
            "{"  # Minimal client context: WEB_REMIX (YT Music Web)
            '"context":{"client":{"clientName":"WEB_REMIX","clientVersion":"1.20250101.00.00"}}'
            "}"
        )
        payload = _http_post(
            url,
            sapisid=self._sid,
            cookies=self._ck,
            origin="https://music.youtube.com",
            timeout=10,
            body=body,
        )
        doc = load_json_str(payload)
        if not isinstance(doc, dict):
            raise AppError(
                code=ErrorCode.EXTERNAL_SERVICE_ERROR, message="invalid yt json", http_status=502
            )
        items = doc.get("items")
        if not isinstance(items, list):
            raise AppError(
                code=ErrorCode.EXTERNAL_SERVICE_ERROR, message="invalid yt payload", http_status=502
            )
        out: list[PlayRecord] = []
        for it in items:
            out.append(_decode_yt_item(it))
            if limit is not None and len(out) >= limit:
                break
        return out


def youtube_client(*, sapisid: str, cookies: str) -> YouTubeMusicProto:
    return _YouTubeClient(sapisid=sapisid, cookies=cookies)


__all__ = [
    "YouTubeHttpPostHook",
    "YouTubeMusicProto",
    "http_post_impl",
    "sapisidhash",
    "youtube_client",
]
