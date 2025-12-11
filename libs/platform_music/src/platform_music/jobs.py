from __future__ import annotations

from typing import Literal, TypedDict

from platform_core.errors import AppError, ErrorCode
from platform_core.job_events import JobDomain, default_events_channel
from platform_core.json_utils import dump_json_str
from platform_core.logging import get_logger
from platform_workers.job_context import JobContext, make_job_context
from platform_workers.redis import RedisStrProto

from platform_music.error_codes import MusicWrappedErrorCode
from platform_music.importers.youtube_takeout import (
    decode_stored_plays,
    static_service_from_plays,
)
from platform_music.services.apple import AppleMusicProto
from platform_music.services.lastfm import LastFmProto
from platform_music.services.protocol import MusicServiceProto
from platform_music.services.spotify import SpotifyProto
from platform_music.services.youtube import YouTubeMusicProto
from platform_music.wrapped import WrappedGenerator


class LastFmCredentials(TypedDict):
    api_key: str
    api_secret: str
    session_key: str


class SpotifyCredentials(TypedDict):
    access_token: str
    refresh_token: str
    expires_in: int | str


class AppleMusicCredentials(TypedDict):
    developer_token: str
    music_user_token: str


class YouTubeMusicCredentials(TypedDict):
    sapisid: str
    cookies: str


ServiceCredentials = (
    LastFmCredentials | SpotifyCredentials | AppleMusicCredentials | YouTubeMusicCredentials
)


class WrappedJobPayload(TypedDict):
    type: Literal["music_wrapped.generate.v1"]
    year: int
    service: Literal["lastfm", "spotify", "apple_music", "youtube_music"]
    credentials: ServiceCredentials
    user_id: int
    redis_url: str
    queue_name: str


_MUSIC_DOMAIN: JobDomain = "music_wrapped"


def _redis_client(url: str) -> RedisStrProto:
    """Get redis client via hook. Hook is always set (production or test fake)."""
    from platform_music.testing import hooks

    return hooks.redis_client(url)


def _get_lastfm_client(*, api_key: str, api_secret: str, session_key: str) -> LastFmProto:
    """Get lastfm client via hook. Hook is always set (production or test fake)."""
    from platform_music.testing import hooks

    return hooks.lastfm_client(api_key, api_secret, session_key)


def _get_spotify_client(
    *, access_token: str, refresh_token: str, expires_in: int | str
) -> SpotifyProto:
    """Get spotify client via hook. Hook is always set (production or test fake)."""
    from platform_music.testing import hooks

    return hooks.spotify_client(access_token, refresh_token, expires_in)


def _get_apple_client(*, music_user_token: str, developer_token: str) -> AppleMusicProto:
    """Get apple client via hook. Hook is always set (production or test fake)."""
    from platform_music.testing import hooks

    return hooks.apple_client(music_user_token, developer_token)


def _get_youtube_client(*, sapisid: str, cookies: str) -> YouTubeMusicProto:
    """Get youtube client via hook. Hook is always set (production or test fake)."""
    from platform_music.testing import hooks

    return hooks.youtube_client(sapisid, cookies)


def _as_lastfm_creds(creds: ServiceCredentials) -> LastFmCredentials:
    raw = dict(creds)
    api_key = raw.get("api_key")
    api_secret = raw.get("api_secret")
    session_key = raw.get("session_key")
    if (
        not isinstance(api_key, str)
        or not isinstance(api_secret, str)
        or not isinstance(session_key, str)
    ):
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_CREDENTIALS,
            message="invalid lastfm credentials",
            http_status=400,
        )
    out: LastFmCredentials = {
        "api_key": api_key,
        "api_secret": api_secret,
        "session_key": session_key,
    }
    return out


def _as_spotify_creds(creds: ServiceCredentials) -> SpotifyCredentials:
    raw = dict(creds)
    at = raw.get("access_token")
    rt = raw.get("refresh_token")
    ex = raw.get("expires_in")
    if not isinstance(at, str) or not isinstance(rt, str) or not isinstance(ex, (int, str)):
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_CREDENTIALS,
            message="invalid spotify credentials",
            http_status=400,
        )
    out: SpotifyCredentials = {"access_token": at, "refresh_token": rt, "expires_in": ex}
    return out


def _as_apple_creds(creds: ServiceCredentials) -> AppleMusicCredentials:
    raw = dict(creds)
    dev = raw.get("developer_token")
    mus = raw.get("music_user_token")
    if not isinstance(dev, str) or not isinstance(mus, str):
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_CREDENTIALS,
            message="invalid apple music credentials",
            http_status=400,
        )
    out: AppleMusicCredentials = {"developer_token": dev, "music_user_token": mus}
    return out


def _as_youtube_creds(creds: ServiceCredentials) -> YouTubeMusicCredentials:
    raw = dict(creds)
    sid = raw.get("sapisid")
    ck = raw.get("cookies")
    if not isinstance(sid, str) or not isinstance(ck, str):
        raise AppError(
            code=MusicWrappedErrorCode.INVALID_CREDENTIALS,
            message="invalid youtube music credentials",
            http_status=400,
        )
    out: YouTubeMusicCredentials = {"sapisid": sid, "cookies": ck}
    return out


def process_wrapped_job(payload: WrappedJobPayload) -> str:
    """Worker job to generate Music Wrapped and store the result in Redis.

    Publishes started/progress/completed/failed events on the music_wrapped domain.
    Returns a result_id used to retrieve the JSON result from Redis.
    """
    year = int(payload["year"])
    service = payload["service"]
    user_id = int(payload["user_id"])
    redis_url = payload["redis_url"]
    queue_name = payload["queue_name"]

    redis = _redis_client(redis_url)
    ctx: JobContext = make_job_context(
        redis=redis,
        domain=_MUSIC_DOMAIN,
        events_channel=default_events_channel(_MUSIC_DOMAIN),
        job_id=f"wrapped-{user_id}-{year}",
        user_id=user_id,
        queue_name=queue_name,
    )
    ctx.publish_started()

    try:
        ctx.publish_progress(10, "auth")
        client: MusicServiceProto

        def _build_lastfm(creds: ServiceCredentials) -> MusicServiceProto:
            c = _as_lastfm_creds(creds)
            return _get_lastfm_client(
                api_key=c["api_key"],
                api_secret=c["api_secret"],
                session_key=c["session_key"],
            )

        def _build_spotify(creds: ServiceCredentials) -> MusicServiceProto:
            c = _as_spotify_creds(creds)
            return _get_spotify_client(
                access_token=c["access_token"],
                refresh_token=c["refresh_token"],
                expires_in=c["expires_in"],
            )

        def _build_apple(creds: ServiceCredentials) -> MusicServiceProto:
            c = _as_apple_creds(creds)
            return _get_apple_client(
                music_user_token=c["music_user_token"],
                developer_token=c["developer_token"],
            )

        def _build_youtube(creds: ServiceCredentials) -> MusicServiceProto:
            c = _as_youtube_creds(creds)
            return _get_youtube_client(sapisid=c["sapisid"], cookies=c["cookies"])

        builders = {
            "lastfm": _build_lastfm,
            "spotify": _build_spotify,
            "apple_music": _build_apple,
            "youtube_music": _build_youtube,
        }
        client = builders[service](payload["credentials"])

        ctx.publish_progress(25, "history")
        generator = WrappedGenerator(client)
        result = generator.generate_wrapped(year=year)

        ctx.publish_progress(75, "persist")
        result_id = f"wrapped:{user_id}:{year}"
        redis.set(result_id, dump_json_str(result))

        ctx.publish_completed(result_id, 0)
        return result_id
    except Exception as exc:
        # Classify error kind: AppError => user, else system
        kind: Literal["user", "system"] = "user" if isinstance(exc, AppError) else "system"
        ctx.publish_failed(kind, str(exc))
        _log = get_logger(__name__)
        _log.exception("music_wrapped job failed: %s", exc)
        raise


__all__ = [
    "AppleMusicCredentials",
    "LastFmCredentials",
    "ServiceCredentials",
    "SpotifyCredentials",
    "WrappedJobPayload",
    "YouTubeMusicCredentials",
    "process_wrapped_job",
]


class ImportYouTubeTakeoutJobPayload(TypedDict):
    type: Literal["music_wrapped.import_youtube_takeout.v1"]
    year: int
    token_id: str
    user_id: int
    redis_url: str
    queue_name: str


def process_import_youtube_takeout(payload: ImportYouTubeTakeoutJobPayload) -> str:
    """Worker job to generate Music Wrapped from previously uploaded Takeout data.

    The API stores parsed PlayRecords in Redis under key
    "ytmusic:takeout:{token_id}". This job retrieves them, filters by year,
    generates the wrapped JSON, stores it in Redis, and publishes job events.
    """
    year = int(payload["year"])
    token_id = payload["token_id"]
    user_id = int(payload["user_id"])
    redis_url = payload["redis_url"]
    queue_name = payload["queue_name"]

    redis = _redis_client(redis_url)
    ctx: JobContext = make_job_context(
        redis=redis,
        domain=_MUSIC_DOMAIN,
        events_channel=default_events_channel(_MUSIC_DOMAIN),
        job_id=f"wrapped-takeout-{user_id}-{year}",
        user_id=user_id,
        queue_name=queue_name,
    )
    ctx.publish_started()

    try:
        ctx.publish_progress(10, "load")
        key = f"ytmusic:takeout:{token_id}"
        raw = redis.get(key)
        if raw is None:
            raise AppError(code=ErrorCode.NOT_FOUND, message="takeout not found", http_status=404)

        from platform_core.json_utils import load_json_str as _load

        doc = _load(raw)
        plays = decode_stored_plays(doc)
        client = static_service_from_plays(plays)

        ctx.publish_progress(25, "history")
        generator = WrappedGenerator(client)
        result = generator.generate_wrapped(year=year)

        ctx.publish_progress(75, "persist")
        result_id = f"wrapped:{user_id}:{year}"
        redis.set(result_id, dump_json_str(result))

        ctx.publish_completed(result_id, 0)
        return result_id
    except Exception as exc:
        kind: Literal["user", "system"] = "user" if isinstance(exc, AppError) else "system"
        ctx.publish_failed(kind, str(exc))
        _log = get_logger(__name__)
        _log.exception("music_wrapped takeout job failed: %s", exc)
        raise


__all__.extend(["ImportYouTubeTakeoutJobPayload", "process_import_youtube_takeout"])
