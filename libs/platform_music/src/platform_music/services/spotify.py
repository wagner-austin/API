from __future__ import annotations

import types
import urllib.parse
from typing import Protocol, runtime_checkable

from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import load_json_str

from platform_music.models import PlayRecord
from platform_music.services.decoders import _decode_spotify_play
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


def _http_get(url: str, *, access_token: str, timeout: float) -> str:
    imp = __import__
    req_mod = imp("urllib.request", fromlist=["Request"])
    req: _RequestObj = req_mod.Request(url)
    req.add_header("Authorization", "Bearer " + access_token)
    opener: _UrlOpen = imp("urllib.request", fromlist=["urlopen"])
    with opener.urlopen(req, timeout=timeout) as resp:
        return resp.read().decode("utf-8")


def _to_unix_seconds(iso: str) -> int:
    from datetime import datetime

    return int(datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp())


@runtime_checkable
class SpotifyProto(MusicServiceProto, Protocol):
    """Protocol for Spotify service implementations."""

    pass


class _SpotifyClient(SpotifyProto):
    def __init__(self, *, access_token: str) -> None:
        self._token = access_token

    def get_listening_history(
        self, *, start_date: str, end_date: str, limit: int | None = None
    ) -> list[PlayRecord]:
        after = _to_unix_seconds(start_date) * 1000
        before = _to_unix_seconds(end_date) * 1000
        url_base = "https://api.spotify.com/v1/me/player/recently-played"
        total: list[PlayRecord] = []
        next_before: int = before
        pages = 0
        max_pages = 5 if limit is None else 50
        while pages < max_pages:
            params: dict[str, str] = {"limit": "50"}
            params["before"] = str(next_before)
            if after > 0:
                params["after"] = str(after)
            url = url_base + "?" + urllib.parse.urlencode(params)
            payload = _http_get(url, access_token=self._token, timeout=10)
            doc = load_json_str(payload)
            if not isinstance(doc, dict):
                raise AppError(
                    code=ErrorCode.EXTERNAL_SERVICE_ERROR,
                    message="invalid spotify json",
                    http_status=502,
                )
            items = doc.get("items")
            if not isinstance(items, list) or len(items) == 0:
                break
            for raw in items:
                rec = _decode_spotify_play(raw)
                total.append(rec)
                if limit is not None and len(total) >= limit:
                    return total
            cursors = doc.get("cursors")
            if isinstance(cursors, dict):
                b = cursors.get("before")
                if isinstance(b, str) and b.isdigit():
                    next_before = int(b)
                else:
                    break
            else:
                break
            pages += 1
        return total


def spotify_client(*, access_token: str, refresh_token: str, expires_in: int | str) -> SpotifyProto:
    _ = (refresh_token, expires_in)
    return _SpotifyClient(access_token=access_token)


__all__ = ["SpotifyProto", "spotify_client"]
__import__ = __import__
