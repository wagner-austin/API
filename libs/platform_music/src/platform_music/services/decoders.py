from __future__ import annotations

from datetime import UTC, datetime

from platform_core.json_utils import JSONValue

from platform_music.models import PlayRecord, Track


class DecoderError(ValueError):
    """Raised when a service JSON payload fails validation."""


def _iso_from_uts(uts: str | int) -> str:
    if isinstance(uts, int):
        ts = uts
    elif isinstance(uts, str) and uts.isdigit():
        ts = int(uts)
    else:
        raise DecoderError("date.uts must be an integer or numeric string")
    return datetime.fromtimestamp(ts, tz=UTC).isoformat().replace("+00:00", "Z")


def _decode_lastfm_scrobble(raw: JSONValue) -> PlayRecord:
    """Decode a Last.fm scrobble JSON object into a PlayRecord.

    Expected minimal shape (keys used):
    {
        "artist": {"#text": "Artist Name"},
        "name": "Track Title",
        "date": {"uts": "1704067200"}
    }
    """
    if not isinstance(raw, dict):
        raise DecoderError("scrobble must be a dict")

    artist_obj = raw.get("artist")
    if not isinstance(artist_obj, dict):
        raise DecoderError("artist must be a dict")
    artist_name_val = artist_obj.get("#text")
    if not isinstance(artist_name_val, str) or artist_name_val == "":
        raise DecoderError("artist.#text must be a non-empty string")
    artist_name = artist_name_val

    name_val = raw.get("name")
    if not isinstance(name_val, str) or name_val == "":
        raise DecoderError("name must be a non-empty string")
    title = name_val

    date_obj = raw.get("date")
    if not isinstance(date_obj, dict):
        raise DecoderError("date must be a dict")
    uts_val = date_obj.get("uts")
    if not isinstance(uts_val, (int, str)):
        raise DecoderError("date.uts must be an integer or numeric string")
    played_at = _iso_from_uts(uts_val)

    # Construct a minimal Track, synthesizing an id from artist+title for Last.fm
    track: Track = {
        "id": f"lastfm:{artist_name}:{title}",
        "title": title,
        "artist_name": artist_name,
        "duration_ms": 0,
        "service": "lastfm",
    }
    record: PlayRecord = {"track": track, "played_at": played_at, "service": "lastfm"}
    return record


def _decode_spotify_play(raw: JSONValue) -> PlayRecord:
    """Decode a Spotify recently-played item into a PlayRecord.

    Expected minimal shape (keys used):
    {
        "track": {
            "id": "spotify-track-id",
            "name": "Track Title",
            "artists": [{"name": "Artist Name"}],
            "duration_ms": 123456
        },
        "played_at": "2024-01-01T00:00:00Z"
    }
    """
    if not isinstance(raw, dict):
        raise DecoderError("spotify item must be a dict")

    track_obj = raw.get("track")
    if not isinstance(track_obj, dict):
        raise DecoderError("track must be a dict")

    tid = track_obj.get("id")
    if not isinstance(tid, str) or tid == "":
        raise DecoderError("track.id must be a non-empty string")

    name_val = track_obj.get("name")
    if not isinstance(name_val, str) or name_val == "":
        raise DecoderError("track.name must be a non-empty string")

    artists_val = track_obj.get("artists")
    if not isinstance(artists_val, list) or len(artists_val) == 0:
        raise DecoderError("track.artists must be a non-empty list")
    first_artist = artists_val[0]
    if not isinstance(first_artist, dict):
        raise DecoderError("track.artists[0] must be a dict")
    artist_name_val = first_artist.get("name")
    if not isinstance(artist_name_val, str) or artist_name_val == "":
        raise DecoderError("track.artists[0].name must be a non-empty string")

    dur_val = track_obj.get("duration_ms")
    if not isinstance(dur_val, int) or dur_val < 0:
        raise DecoderError("track.duration_ms must be a non-negative int")

    played_at_val = raw.get("played_at")
    if not isinstance(played_at_val, str) or played_at_val == "":
        raise DecoderError("played_at must be a non-empty string")

    track: Track = {
        "id": tid,
        "title": name_val,
        "artist_name": artist_name_val,
        "duration_ms": dur_val,
        "service": "spotify",
    }
    record: PlayRecord = {"track": track, "played_at": played_at_val, "service": "spotify"}
    return record


__all__ = [
    "DecoderError",
    "_decode_lastfm_scrobble",
    "_decode_spotify_play",
]
