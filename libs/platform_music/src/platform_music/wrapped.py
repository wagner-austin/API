from __future__ import annotations

from collections import Counter
from datetime import UTC, datetime

from platform_core.errors import AppError
from platform_core.json_utils import JSONValue
from platform_core.validators import (
    load_json_dict,
    validate_int_range,
    validate_required_literal,
    validate_str,
)

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

_SERVICE_VALUES = frozenset({"lastfm", "spotify", "apple_music", "youtube_music"})
_SERVICE_MAP: dict[str, ServiceName] = {
    "lastfm": "lastfm",
    "spotify": "spotify",
    "apple_music": "apple_music",
    "youtube_music": "youtube_music",
}


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
    d = load_json_dict(
        doc,
        error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
        message="object required",
        http_status=400,
    )

    service_val = validate_required_literal(
        d.get("service"),
        "service",
        _SERVICE_VALUES,
        error_code=MusicWrappedErrorCode.INVALID_SERVICE,
        http_status=400,
    )
    service_name = _SERVICE_MAP[service_val]

    year = validate_int_range(
        d.get("year"),
        "year",
        error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
        http_status=400,
    )

    generated_at = validate_str(
        d.get("generated_at"),
        "generated_at",
        error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
        http_status=400,
    )
    if not generated_at:
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            message="generated_at required",
            http_status=400,
        )

    total_scrobbles = validate_int_range(
        d.get("total_scrobbles"),
        "total_scrobbles",
        ge=0,
        error_code=MusicWrappedErrorCode.INSUFFICIENT_DATA,
        http_status=400,
    )

    artists = _validate_top_artists(d.get("top_artists"))
    songs = _validate_top_songs(d.get("top_songs"))
    by_month = _validate_top_by_month(d.get("top_by_month"))

    out: WrappedResult = {
        "service": service_name,
        "year": year,
        "generated_at": generated_at,
        "total_scrobbles": total_scrobbles,
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
        d = load_json_dict(
            a,
            error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            message="invalid artist entry",
            http_status=400,
        )
        name = validate_str(
            d.get("artist_name"),
            "artist_name",
            error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            http_status=400,
        )
        plays = validate_int_range(
            d.get("play_count"),
            "play_count",
            error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            http_status=400,
        )
        out.append({"artist_name": name, "play_count": plays})
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
        d = load_json_dict(
            s,
            error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            message="invalid song entry",
            http_status=400,
        )
        title = validate_str(
            d.get("title"),
            "title",
            error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            http_status=400,
        )
        artist = validate_str(
            d.get("artist_name"),
            "artist_name",
            error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            http_status=400,
        )
        plays = validate_int_range(
            d.get("play_count"),
            "play_count",
            error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            http_status=400,
        )
        out.append({"title": title, "artist_name": artist, "play_count": plays})
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
        d = load_json_dict(
            e,
            error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            message="invalid month entry",
            http_status=400,
        )
        month = validate_int_range(
            d.get("month"),
            "month",
            ge=1,
            le=12,
            error_code=MusicWrappedErrorCode.INVALID_DATE_RANGE,
            http_status=400,
        )
        top_artists = _validate_top_artists(d.get("top_artists"))
        out.append({"month": month, "top_artists": top_artists})
    return out
