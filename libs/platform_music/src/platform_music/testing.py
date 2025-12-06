from __future__ import annotations

from platform_music.models import PlayRecord, Track
from platform_music.services.lastfm import LastFmProto


class FakeLastFm(LastFmProto):
    """Simple in-memory Last.fm fake for testing."""

    def __init__(self) -> None:
        self._plays: list[PlayRecord] = []

    def add_play(
        self,
        *,
        track_id: str,
        title: str,
        artist_name: str,
        played_at: str,
    ) -> None:
        track: Track = {
            "id": track_id,
            "title": title,
            "artist_name": artist_name,
            "duration_ms": 0,
            "service": "lastfm",
        }
        self._plays.append({"track": track, "played_at": played_at, "service": "lastfm"})

    def get_listening_history(
        self,
        *,
        start_date: str,
        end_date: str,
        limit: int | None = None,
    ) -> list[PlayRecord]:
        # ISO 8601 string compare is valid for UTC timestamps
        filtered = [p for p in self._plays if start_date <= p["played_at"] <= end_date]
        if limit is not None:
            filtered = filtered[:limit]
        return list(filtered)
