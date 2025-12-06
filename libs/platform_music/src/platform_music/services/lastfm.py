from __future__ import annotations

from typing import Protocol, runtime_checkable

from platform_music.models import PlayRecord
from platform_music.services.protocol import MusicServiceProto


@runtime_checkable
class LastFmProto(MusicServiceProto, Protocol):
    """Marker Protocol for Last.fm service implementations."""

    pass


def lastfm_client(api_key: str, api_secret: str, session_key: str) -> LastFmProto:
    """Dynamic factory for Last.fm client.

    Tests monkeypatch this to return a FakeLastFm. The default implementation
    uses a stub to avoid accidental runtime behavior in strict tests.
    """
    _ = (api_key, api_secret, session_key)

    class _Stub(LastFmProto):
        def get_listening_history(
            self, *, start_date: str, end_date: str, limit: int | None = None
        ) -> list[PlayRecord]:
            raise NotImplementedError("lastfm_client requires a concrete adapter in runtime")

    return _Stub()
