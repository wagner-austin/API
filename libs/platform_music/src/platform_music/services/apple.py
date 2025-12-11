from __future__ import annotations

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
AppleHttpGetHook = Callable[[str, str, str, float], str]


def http_get_impl(url: str, *, developer_token: str, user_token: str, timeout: float) -> str:
    """Production HTTP GET implementation for Apple Music API.

    Uses urllib.request to make HTTP GET requests with Bearer token and Music-User-Token.
    This function is exported for use by testing.py to set production hooks.
    """
    imp = __import__
    req_mod = imp("urllib.request", fromlist=["Request"])
    req: _RequestObj = req_mod.Request(url)
    req.add_header("Authorization", "Bearer " + developer_token)
    req.add_header("Music-User-Token", user_token)
    opener: _UrlOpen = imp("urllib.request", fromlist=["urlopen"])
    with opener.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _http_get(url: str, *, developer_token: str, user_token: str, timeout: float) -> str:
    """HTTP GET via hook. Hook is always set (production or test fake)."""
    from platform_music.testing import hooks

    return hooks.apple_http_get(url, developer_token, user_token, timeout)


def _decode_apple_item(raw: JSONValue) -> PlayRecord:
    if not isinstance(raw, dict):
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR, message="invalid apple item", http_status=502
        )
    attrs = raw.get("attributes")
    if not isinstance(attrs, dict):
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR, message="missing attributes", http_status=502
        )
    title = attrs.get("name")
    artist = attrs.get("artistName")
    dur = attrs.get("durationInMillis")
    played = attrs.get("lastPlayedDate")
    if not (
        isinstance(title, str)
        and isinstance(artist, str)
        and isinstance(dur, int)
        and isinstance(played, str)
    ):
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR, message="invalid attributes", http_status=502
        )
    tid = raw.get("id")
    if not isinstance(tid, str):
        tid = "apple:" + title
    track: Track = {
        "id": tid,
        "title": title,
        "artist_name": artist,
        "duration_ms": int(dur),
        "service": "apple_music",
    }
    return {"track": track, "played_at": played, "service": "apple_music"}


@runtime_checkable
class AppleMusicProto(MusicServiceProto, Protocol):
    pass


class _AppleClient(AppleMusicProto):
    def __init__(self, *, developer_token: str, user_token: str) -> None:
        self._dev = developer_token
        self._usr = user_token

    def get_listening_history(
        self, *, start_date: str, end_date: str, limit: int | None = None
    ) -> list[PlayRecord]:
        # Apple recently played does not filter by date; we fetch recent and the
        # caller constrains via limit.
        url = "https://api.music.apple.com/v1/me/recent/played/tracks?limit=" + str(limit or 25)
        payload = _http_get(url, developer_token=self._dev, user_token=self._usr, timeout=10)
        doc = load_json_str(payload)
        if not isinstance(doc, dict):
            raise AppError(
                code=ErrorCode.EXTERNAL_SERVICE_ERROR, message="invalid apple json", http_status=502
            )
        data = doc.get("data")
        if not isinstance(data, list):
            raise AppError(
                code=ErrorCode.EXTERNAL_SERVICE_ERROR,
                message="invalid apple payload",
                http_status=502,
            )
        out: list[PlayRecord] = []
        for item in data:
            out.append(_decode_apple_item(item))
            if limit is not None and len(out) >= limit:
                break
        return out


def apple_client(*, music_user_token: str, developer_token: str) -> AppleMusicProto:
    return _AppleClient(developer_token=developer_token, user_token=music_user_token)


__all__ = ["AppleHttpGetHook", "AppleMusicProto", "apple_client", "http_get_impl"]
