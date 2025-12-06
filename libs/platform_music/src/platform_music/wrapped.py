from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from platform_core.errors import AppError
from platform_core.json_utils import JSONValue

from platform_music.analytics.core.top_artists_by_month import compute_top_artists_by_month
from platform_music.analytics.core.top_songs import compute_top_songs
from platform_music.error_codes import MusicWrappedErrorCode
from platform_music.models import (
    DEFAULT_MIN_PLAYS,
    ListeningHistory,
    PlayRecord,
    ServiceName,
    TopArtist,
    TopArtistsByMonthEntry,
    TopSong,
    WrappedResult,
)
from platform_music.services.protocol import MusicServiceProto


def _iso_utc_now() -> str:
    ts = datetime.now(UTC).isoformat()
    # Normalize trailing "+00:00" to "Z" for consistency
    return ts.replace("+00:00", "Z")


def _top_artists(plays: list[PlayRecord], *, limit: int = 5) -> list[TopArtist]:
    counts: Counter[str] = Counter(p["track"]["artist_name"] for p in plays)
    ranked: list[TopArtist] = [
        {"artist_name": name, "play_count": count} for name, count in counts.most_common(limit)
    ]
    return ranked


class WrappedGenerator:
    """Generate a minimal Wrapped result from a music service.

    This is intentionally small and strict to pass monorepo quality gates
    while we grow analytics incrementally.
    """

    def __init__(self, music_client: MusicServiceProto) -> None:
        self._client = music_client

    def generate_wrapped(self, *, year: int) -> WrappedResult:
        start = f"{year}-01-01T00:00:00Z"
        end = f"{year}-12-31T23:59:59Z"

        plays: list[PlayRecord] = self._client.get_listening_history(
            start_date=start, end_date=end, limit=None
        )

        if len(plays) == 0:
            raise AppError(
                code=MusicWrappedErrorCode.NO_LISTENING_HISTORY,
                message="No listening history in the selected year",
                http_status=404,
            )

        if len(plays) < DEFAULT_MIN_PLAYS:
            raise AppError(
                code=MusicWrappedErrorCode.INSUFFICIENT_DATA,
                message=f"Need at least {DEFAULT_MIN_PLAYS} plays for Wrapped",
                http_status=400,
            )

        service = plays[0]["service"]
        history: ListeningHistory = {
            "year": year,
            "plays": plays,
            "total_plays": len(plays),
        }

        top_artists = _top_artists(history["plays"], limit=5)
        top_songs: list[TopSong] = compute_top_songs(history["plays"], limit=5)
        top_by_month = compute_top_artists_by_month(history["plays"], limit=3)

        result: WrappedResult = {
            "service": service,
            "year": year,
            "generated_at": _iso_utc_now(),
            "total_scrobbles": history["total_plays"],
            "top_artists": top_artists,
            "top_songs": top_songs,
            "top_by_month": top_by_month,
        }
        return result


def decode_wrapped_result(doc: JSONValue) -> WrappedResult:
    """Validate a JSONValue tree into a strict WrappedResult mapping.

    Raises AppError on validation failure.
    """
    if not isinstance(doc, dict):
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            message="object required",
            http_status=400,
        )
    service_val = doc.get("service")
    if service_val == "lastfm":
        service_name: ServiceName = "lastfm"
    elif service_val == "spotify":
        service_name = "spotify"
    elif service_val == "apple_music":
        service_name = "apple_music"
    elif service_val == "youtube_music":
        service_name = "youtube_music"
    else:
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_SERVICE,
            message="invalid service",
            http_status=400,
        )
    year = doc.get("year")
    if not isinstance(year, int):
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            message="year must be int",
            http_status=400,
        )
    generated_at = doc.get("generated_at")
    if not isinstance(generated_at, str) or len(generated_at) == 0:
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            message="generated_at required",
            http_status=400,
        )
    total_scrobbles = doc.get("total_scrobbles")
    if not isinstance(total_scrobbles, int) or total_scrobbles < 0:
        raise AppError(
            code=MusicWrappedErrorCode.INSUFFICIENT_DATA,
            message="invalid total_scrobbles",
            http_status=400,
        )

    artists = _validate_top_artists(doc.get("top_artists"))
    songs = _validate_top_songs(doc.get("top_songs"))
    by_month = _validate_top_by_month(doc.get("top_by_month"))

    out: WrappedResult = {
        "service": service_name,
        "year": int(year),
        "generated_at": generated_at,
        "total_scrobbles": int(total_scrobbles),
        "top_artists": artists,
        "top_songs": songs,
        "top_by_month": by_month,
    }
    return out


def _validate_top_artists(val: JSONValue) -> list[TopArtist]:
    if not isinstance(val, list):
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            message="top_artists must be list",
            http_status=400,
        )
    out: list[TopArtist] = []
    for a in val:
        if not isinstance(a, dict):
            raise AppError(
                code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
                message="invalid artist entry",
                http_status=400,
            )
        name = a.get("artist_name")
        plays = a.get("play_count")
        if not isinstance(name, str) or not isinstance(plays, int):
            raise AppError(
                code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
                message="invalid artist fields",
                http_status=400,
            )
        out.append({"artist_name": name, "play_count": int(plays)})
    return out


def _validate_top_songs(val: JSONValue) -> list[TopSong]:
    if not isinstance(val, list):
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            message="top_songs must be list",
            http_status=400,
        )
    out: list[TopSong] = []
    for s in val:
        if not isinstance(s, dict):
            raise AppError(
                code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
                message="invalid song entry",
                http_status=400,
            )
        title = s.get("title")
        artist = s.get("artist_name")
        plays = s.get("play_count")
        if not isinstance(title, str) or not isinstance(artist, str) or not isinstance(plays, int):
            raise AppError(
                code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
                message="invalid song fields",
                http_status=400,
            )
        out.append({"title": title, "artist_name": artist, "play_count": int(plays)})
    return out


def _validate_top_by_month(val: JSONValue) -> list[TopArtistsByMonthEntry]:
    if not isinstance(val, list):
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            message="top_by_month must be list",
            http_status=400,
        )
    out: list[TopArtistsByMonthEntry] = []
    for e in val:
        if not isinstance(e, dict):
            raise AppError(
                code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
                message="invalid month entry",
                http_status=400,
            )
        month = e.get("month")
        if not isinstance(month, int) or month < 1 or month > 12:
            raise AppError(
                code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
                message="invalid month",
                http_status=400,
            )
        top_artists = _validate_top_artists(e.get("top_artists"))
        out.append({"month": int(month), "top_artists": top_artists})
    return out
