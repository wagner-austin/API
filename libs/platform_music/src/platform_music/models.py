from __future__ import annotations

from typing import Final, Literal, TypedDict

ServiceName = Literal["lastfm", "spotify", "apple_music", "youtube_music"]


class Track(TypedDict):
    id: str
    title: str
    artist_name: str
    duration_ms: int
    service: ServiceName


class PlayRecord(TypedDict):
    track: Track
    played_at: str  # ISO 8601 timestamp
    service: ServiceName


class TopArtist(TypedDict):
    artist_name: str
    play_count: int


class TopSong(TypedDict):
    title: str
    artist_name: str
    play_count: int


class ListeningHistory(TypedDict):
    year: int
    plays: list[PlayRecord]
    total_plays: int


class WrappedResult(TypedDict):
    service: ServiceName
    year: int
    generated_at: str
    total_scrobbles: int
    top_artists: list[TopArtist]
    top_songs: list[TopSong]
    top_by_month: list[TopArtistsByMonthEntry]


class TopArtistsByMonthEntry(TypedDict):
    month: int
    top_artists: list[TopArtist]


DEFAULT_MIN_PLAYS: Final[int] = 10
