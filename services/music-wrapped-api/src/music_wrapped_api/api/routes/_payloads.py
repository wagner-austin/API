"""Per-service job-payload builders for the wrapped generate routes.

Each ``_payload_*`` / ``_build_*_payload`` function turns one service's
decoded generate request into the queue payload
``platform_music.jobs.process_wrapped_job`` consumes;
:func:`build_payload_for_service` dispatches on the ``service`` field.
The HTTP routes themselves live in :mod:`.wrapped`.
"""

from __future__ import annotations

from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue
from platform_core.queues import MUSIC_WRAPPED_QUEUE
from platform_music.jobs import LastFmCredentials

from music_wrapped_api import _test_hooks

from ._decoders import (
    AppleGenerateFull,
    AppleGenerateToken,
    LastFmGenerate,
    SpotifyGenerateFull,
    SpotifyGenerateToken,
    YouTubeGenerateFull,
    YouTubeGenerateToken,
    _LastFmCredsFull,
    _LastFmCredsSessionOnly,
    decode_wrapped_generate,
    is_full_lastfm_credentials,
    to_full_lastfm_credentials,
)


def _payload_lastfm(req_l: LastFmGenerate, *, redis_url: str) -> dict[str, JSONValue]:
    creds_in: _LastFmCredsFull | _LastFmCredsSessionOnly = req_l["credentials"]
    if is_full_lastfm_credentials(creds_in):
        raw = dict(creds_in)
        lfm: LastFmCredentials = {
            "api_key": str(raw["api_key"]),
            "api_secret": str(raw["api_secret"]),
            "session_key": str(raw["session_key"]),
        }
    else:
        lfm = to_full_lastfm_credentials(
            creds_in,
            api_key_env=_test_hooks.require_env("LASTFM_API_KEY"),
            api_secret_env=_test_hooks.require_env("LASTFM_API_SECRET"),
        )
    creds_json: dict[str, JSONValue] = {
        "api_key": lfm["api_key"],
        "api_secret": lfm["api_secret"],
        "session_key": lfm["session_key"],
    }
    return {
        "type": "music_wrapped.generate.v1",
        "year": req_l["year"],
        "service": "lastfm",
        "credentials": creds_json,
        "user_id": 0,
        "redis_url": redis_url,
        "queue_name": MUSIC_WRAPPED_QUEUE,
    }


def _payload_spotify_token(req_sp: SpotifyGenerateToken, *, redis_url: str) -> dict[str, JSONValue]:
    data = _test_hooks.redis_factory(redis_url).hgetall(
        f"spotify:session:{req_sp['credentials']['token_id']}"
    )
    at, rt, ex = data.get("access_token"), data.get("refresh_token"), data.get("expires_in")
    if not (isinstance(at, str) and isinstance(rt, str) and isinstance(ex, str)):
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="spotify token not found",
            http_status=404,
        )
    creds_json: dict[str, JSONValue] = {
        "access_token": at,
        "refresh_token": rt,
        "expires_in": ex,
    }
    return {
        "type": "music_wrapped.generate.v1",
        "year": int(req_sp["year"]),
        "service": "spotify",
        "credentials": creds_json,
        "user_id": 0,
        "redis_url": redis_url,
        "queue_name": MUSIC_WRAPPED_QUEUE,
    }


def _payload_spotify_full(req_sf: SpotifyGenerateFull, *, redis_url: str) -> dict[str, JSONValue]:
    sc = req_sf["credentials"]
    creds_json: dict[str, JSONValue] = {
        "access_token": sc["access_token"],
        "refresh_token": sc["refresh_token"],
        "expires_in": sc["expires_in"],
    }
    return {
        "type": "music_wrapped.generate.v1",
        "year": int(req_sf["year"]),
        "service": "spotify",
        "credentials": creds_json,
        "user_id": 0,
        "redis_url": redis_url,
        "queue_name": MUSIC_WRAPPED_QUEUE,
    }


def _payload_apple_token(req_ap: AppleGenerateToken, *, redis_url: str) -> dict[str, JSONValue]:
    data2 = _test_hooks.redis_factory(redis_url).hgetall(
        f"apple:session:{req_ap['credentials']['token_id']}"
    )
    mus = data2.get("music_user_token")
    if not isinstance(mus, str):
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="apple token not found",
            http_status=404,
        )
    creds_json: dict[str, JSONValue] = {
        "music_user_token": mus,
        "developer_token": _test_hooks.require_env("APPLE_DEVELOPER_TOKEN"),
    }
    return {
        "type": "music_wrapped.generate.v1",
        "year": int(req_ap["year"]),
        "service": "apple_music",
        "credentials": creds_json,
        "user_id": 0,
        "redis_url": redis_url,
        "queue_name": MUSIC_WRAPPED_QUEUE,
    }


def _payload_apple_full(req_af: AppleGenerateFull, *, redis_url: str) -> dict[str, JSONValue]:
    ac = req_af["credentials"]
    creds_json: dict[str, JSONValue] = {
        "music_user_token": ac["music_user_token"],
        "developer_token": ac["developer_token"],
    }
    return {
        "type": "music_wrapped.generate.v1",
        "year": int(req_af["year"]),
        "service": "apple_music",
        "credentials": creds_json,
        "user_id": 0,
        "redis_url": redis_url,
        "queue_name": MUSIC_WRAPPED_QUEUE,
    }


def _payload_youtube_token(req_yt: YouTubeGenerateToken, *, redis_url: str) -> dict[str, JSONValue]:
    data3 = _test_hooks.redis_factory(redis_url).hgetall(
        f"ytmusic:session:{req_yt['credentials']['token_id']}"
    )
    sid, ck = data3.get("sapisid"), data3.get("cookies")
    if not isinstance(sid, str) or not isinstance(ck, str):
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="youtube token not found",
            http_status=404,
        )
    creds_json: dict[str, JSONValue] = {"sapisid": sid, "cookies": ck}
    return {
        "type": "music_wrapped.generate.v1",
        "year": int(req_yt["year"]),
        "service": "youtube_music",
        "credentials": creds_json,
        "user_id": 0,
        "redis_url": redis_url,
        "queue_name": MUSIC_WRAPPED_QUEUE,
    }


def _payload_youtube_full(req_yf: YouTubeGenerateFull, *, redis_url: str) -> dict[str, JSONValue]:
    yc = req_yf["credentials"]
    creds_json: dict[str, JSONValue] = {"sapisid": yc["sapisid"], "cookies": yc["cookies"]}
    return {
        "type": "music_wrapped.generate.v1",
        "year": int(req_yf["year"]),
        "service": "youtube_music",
        "credentials": creds_json,
        "user_id": 0,
        "redis_url": redis_url,
        "queue_name": MUSIC_WRAPPED_QUEUE,
    }


def _doc_year(doc: dict[str, JSONValue]) -> int:
    y = doc.get("year")
    if not isinstance(y, int):
        raise AppError(code=ErrorCode.INVALID_INPUT, message="year must be int", http_status=400)
    return int(y)


def _build_spotify_payload(doc: dict[str, JSONValue], *, redis_url: str) -> dict[str, JSONValue]:
    cred = doc.get("credentials")
    if isinstance(cred, dict) and "token_id" in cred and isinstance(cred["token_id"], str):
        req_tok: SpotifyGenerateToken = {
            "year": _doc_year(doc),
            "service": "spotify",
            "credentials": {"token_id": cred["token_id"]},
        }
        return _payload_spotify_token(req_tok, redis_url=redis_url)
    if isinstance(cred, dict):
        at = cred.get("access_token")
        rt = cred.get("refresh_token")
        ex = cred.get("expires_in")
        if not isinstance(at, str) or not isinstance(rt, str) or not isinstance(ex, int):
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message="invalid spotify credentials",
                http_status=400,
            )
        req_full: SpotifyGenerateFull = {
            "year": _doc_year(doc),
            "service": "spotify",
            "credentials": {"access_token": at, "refresh_token": rt, "expires_in": ex},
        }
        return _payload_spotify_full(req_full, redis_url=redis_url)
    raise AppError(code=ErrorCode.INVALID_INPUT, message="invalid spotify payload", http_status=400)


def _build_apple_payload(doc: dict[str, JSONValue], *, redis_url: str) -> dict[str, JSONValue]:
    cred = doc.get("credentials")
    if isinstance(cred, dict) and "token_id" in cred and isinstance(cred["token_id"], str):
        req_ap: AppleGenerateToken = {
            "year": _doc_year(doc),
            "service": "apple_music",
            "credentials": {"token_id": cred["token_id"]},
        }
        return _payload_apple_token(req_ap, redis_url=redis_url)
    if isinstance(cred, dict):
        mus = cred.get("music_user_token")
        dev = cred.get("developer_token")
        if not isinstance(mus, str) or not isinstance(dev, str):
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message="invalid apple credentials",
                http_status=400,
            )
        req_af: AppleGenerateFull = {
            "year": _doc_year(doc),
            "service": "apple_music",
            "credentials": {"music_user_token": mus, "developer_token": dev},
        }
        return _payload_apple_full(req_af, redis_url=redis_url)
    raise AppError(code=ErrorCode.INVALID_INPUT, message="invalid apple payload", http_status=400)


def _build_youtube_payload(doc: dict[str, JSONValue], *, redis_url: str) -> dict[str, JSONValue]:
    cred = doc.get("credentials")
    if isinstance(cred, dict) and "token_id" in cred and isinstance(cred["token_id"], str):
        req_yt: YouTubeGenerateToken = {
            "year": _doc_year(doc),
            "service": "youtube_music",
            "credentials": {"token_id": cred["token_id"]},
        }
        return _payload_youtube_token(req_yt, redis_url=redis_url)
    if isinstance(cred, dict):
        sid = cred.get("sapisid")
        ck = cred.get("cookies")
        if not isinstance(sid, str) or not isinstance(ck, str):
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message="invalid youtube credentials",
                http_status=400,
            )
        req_yf: YouTubeGenerateFull = {
            "year": _doc_year(doc),
            "service": "youtube_music",
            "credentials": {"sapisid": sid, "cookies": ck},
        }
        return _payload_youtube_full(req_yf, redis_url=redis_url)
    raise AppError(code=ErrorCode.INVALID_INPUT, message="invalid youtube payload", http_status=400)


def build_payload_for_service(doc: JSONValue, *, redis_url: str) -> dict[str, JSONValue]:
    if not isinstance(doc, dict):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="object body required",
            http_status=400,
        )
    svc_val = doc.get("service")
    if not isinstance(svc_val, str):
        raise AppError(code=ErrorCode.INVALID_INPUT, message="service required", http_status=400)
    svc = svc_val
    if svc == "lastfm":
        return _payload_lastfm(decode_wrapped_generate(doc), redis_url=redis_url)
    if svc == "spotify":
        return _build_spotify_payload(doc, redis_url=redis_url)
    if svc == "apple_music":
        return _build_apple_payload(doc, redis_url=redis_url)
    if svc == "youtube_music":
        return _build_youtube_payload(doc, redis_url=redis_url)
    raise AppError(code=ErrorCode.INVALID_INPUT, message="unsupported service", http_status=400)


__all__ = ["build_payload_for_service"]
