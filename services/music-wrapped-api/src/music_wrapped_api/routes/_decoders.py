from __future__ import annotations

from typing import Literal, TypedDict, TypeGuard

from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue
from platform_music.jobs import (
    AppleMusicCredentials,
    LastFmCredentials,
    SpotifyCredentials,
    YouTubeMusicCredentials,
)


class _LastFmCredsFull(TypedDict):
    api_key: str
    api_secret: str
    session_key: str


class _LastFmCredsSessionOnly(TypedDict):
    session_key: str


# Note: avoid TypeAlias; use explicit unions at call sites


class GenerateRequest(TypedDict):
    year: int
    service: Literal["lastfm"]
    credentials: _LastFmCredsFull | _LastFmCredsSessionOnly


def decode_wrapped_generate(doc: JSONValue) -> GenerateRequest:
    """Decode and validate the generate request body into a strict mapping.

    - object with fields: year: int, service: "lastfm",
      credentials with keys api_key, api_secret, session_key as str.
    - raises AppError(ErrorCode.INVALID_INPUT, 400) on validation failure.
    """
    if not isinstance(doc, dict):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="object body required",
            http_status=400,
        )

    year_val = doc.get("year")
    if not isinstance(year_val, int):
        raise AppError(code=ErrorCode.INVALID_INPUT, message="year must be int", http_status=400)

    service_val = doc.get("service")
    if service_val != "lastfm":
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="service must be 'lastfm'",
            http_status=400,
        )

    creds_val = doc.get("credentials")
    if not isinstance(creds_val, dict):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="credentials must be object",
            http_status=400,
        )

    # Accept either full creds or session-only input. If api_key/api_secret keys are present,
    # enforce they are strings.
    api_key_val = creds_val.get("api_key")
    api_secret_val = creds_val.get("api_secret")
    session_key_val = creds_val.get("session_key")
    has_keys = ("api_key" in creds_val) or ("api_secret" in creds_val)
    if has_keys:
        if not (isinstance(api_key_val, str) and isinstance(api_secret_val, str)):
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message="credentials.api_key and api_secret must be str",
                http_status=400,
            )
        if not isinstance(session_key_val, str):
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message="credentials.session_key must be str",
                http_status=400,
            )
        creds: _LastFmCredsFull | _LastFmCredsSessionOnly = {
            "api_key": api_key_val,
            "api_secret": api_secret_val,
            "session_key": session_key_val,
        }
    else:
        if not isinstance(session_key_val, str):
            raise AppError(
                code=ErrorCode.INVALID_INPUT,
                message="credentials.session_key must be str",
                http_status=400,
            )
        creds = {"session_key": session_key_val}
    out: GenerateRequest = {
        "year": int(year_val),
        "service": "lastfm",
        "credentials": creds,
    }
    return out


__all__ = [
    "GenerateRequest",
    "LastFmGenerate",
    "decode_apple_credentials",
    "decode_apple_store",
    "decode_spotify_credentials",
    "decode_wrapped_generate",
    "decode_youtube_credentials",
    "is_full_lastfm_credentials",
    "to_full_lastfm_credentials",
]

# Back-compat alias for clearer naming at call sites
LastFmGenerate = GenerateRequest


class _TokenRef(TypedDict):
    token_id: str


class SpotifyGenerateToken(TypedDict):
    year: int
    service: Literal["spotify"]
    credentials: _TokenRef


class SpotifyGenerateFull(TypedDict):
    year: int
    service: Literal["spotify"]
    credentials: SpotifyCredentials


class AppleGenerateToken(TypedDict):
    year: int
    service: Literal["apple_music"]
    credentials: _TokenRef


class AppleGenerateFull(TypedDict):
    year: int
    service: Literal["apple_music"]
    credentials: AppleMusicCredentials


class YouTubeGenerateToken(TypedDict):
    year: int
    service: Literal["youtube_music"]
    credentials: _TokenRef


class YouTubeGenerateFull(TypedDict):
    year: int
    service: Literal["youtube_music"]
    credentials: YouTubeMusicCredentials


# Avoid TypeAlias; expand unions explicitly in annotations


def _decode_token_ref(doc: JSONValue) -> _TokenRef:
    if not isinstance(doc, dict):
        raise AppError(code=ErrorCode.INVALID_INPUT, message="object required", http_status=400)
    tid = doc.get("token_id")
    if not isinstance(tid, str):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="token_id must be str",
            http_status=400,
        )
    return {"token_id": tid}


def decode_generate_any(
    doc: JSONValue,
) -> (
    GenerateRequest
    | SpotifyGenerateToken
    | SpotifyGenerateFull
    | AppleGenerateToken
    | AppleGenerateFull
    | YouTubeGenerateToken
    | YouTubeGenerateFull
):
    if not isinstance(doc, dict):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="object body required",
            http_status=400,
        )
    year_val = doc.get("year")
    if not isinstance(year_val, int):
        raise AppError(code=ErrorCode.INVALID_INPUT, message="year must be int", http_status=400)
    svc = doc.get("service")
    if svc == "lastfm":
        return decode_wrapped_generate(doc)
    creds_val = doc.get("credentials")
    if svc == "spotify":
        if isinstance(creds_val, dict) and "token_id" in creds_val:
            ref = _decode_token_ref(creds_val)
            return {"year": int(year_val), "service": "spotify", "credentials": ref}
        full = decode_spotify_credentials(creds_val)
        return {"year": int(year_val), "service": "spotify", "credentials": full}
    if svc == "apple_music":
        if isinstance(creds_val, dict) and "token_id" in creds_val:
            ref2 = _decode_token_ref(creds_val)
            return {"year": int(year_val), "service": "apple_music", "credentials": ref2}
        full2 = decode_apple_credentials(creds_val)
        return {"year": int(year_val), "service": "apple_music", "credentials": full2}
    if svc == "youtube_music":
        if isinstance(creds_val, dict) and "token_id" in creds_val:
            ref3 = _decode_token_ref(creds_val)
            return {"year": int(year_val), "service": "youtube_music", "credentials": ref3}
        full3 = decode_youtube_credentials(creds_val)
        return {"year": int(year_val), "service": "youtube_music", "credentials": full3}
    raise AppError(code=ErrorCode.INVALID_INPUT, message="unsupported service", http_status=400)


def to_full_lastfm_credentials(
    creds: _LastFmCredsFull | _LastFmCredsSessionOnly, *, api_key_env: str, api_secret_env: str
) -> LastFmCredentials:
    sk = creds["session_key"]
    if is_full_lastfm_credentials(creds):
        return {"api_key": creds["api_key"], "api_secret": creds["api_secret"], "session_key": sk}
    return {"api_key": api_key_env, "api_secret": api_secret_env, "session_key": sk}


def is_full_lastfm_credentials(
    creds: _LastFmCredsFull | _LastFmCredsSessionOnly,
) -> TypeGuard[_LastFmCredsFull]:
    return ("api_key" in creds) and ("api_secret" in creds)


def decode_spotify_credentials(doc: JSONValue) -> SpotifyCredentials:
    if not isinstance(doc, dict):
        raise AppError(code=ErrorCode.INVALID_INPUT, message="object required", http_status=400)
    at = doc.get("access_token")
    rt = doc.get("refresh_token")
    ex = doc.get("expires_in")
    if not isinstance(at, str) or not isinstance(rt, str) or not isinstance(ex, int):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="invalid spotify credentials",
            http_status=400,
        )
    return {"access_token": at, "refresh_token": rt, "expires_in": ex}


def decode_apple_credentials(doc: JSONValue) -> AppleMusicCredentials:
    if not isinstance(doc, dict):
        raise AppError(code=ErrorCode.INVALID_INPUT, message="object required", http_status=400)
    dev = doc.get("developer_token")
    mus = doc.get("music_user_token")
    if not isinstance(dev, str) or not isinstance(mus, str):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="invalid apple music credentials",
            http_status=400,
        )
    return {"developer_token": dev, "music_user_token": mus}


def decode_youtube_credentials(doc: JSONValue) -> YouTubeMusicCredentials:
    if not isinstance(doc, dict):
        raise AppError(code=ErrorCode.INVALID_INPUT, message="object required", http_status=400)
    sid = doc.get("sapisid")
    ck = doc.get("cookies")
    if not isinstance(sid, str) or not isinstance(ck, str):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="invalid youtube_music credentials",
            http_status=400,
        )
    return {"sapisid": sid, "cookies": ck}


class AppleStoreInput(TypedDict):
    music_user_token: str


def decode_apple_store(doc: JSONValue) -> AppleStoreInput:
    if not isinstance(doc, dict):
        raise AppError(code=ErrorCode.INVALID_INPUT, message="object required", http_status=400)
    mus = doc.get("music_user_token")
    if not isinstance(mus, str):
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="invalid apple store input",
            http_status=400,
        )
    return {"music_user_token": mus}
