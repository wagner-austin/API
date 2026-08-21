from __future__ import annotations

import hashlib
import urllib.parse
from typing import Annotated

from fastapi import APIRouter, File, Request, UploadFile
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue, load_json_bytes
from platform_core.queues import MUSIC_WRAPPED_QUEUE
from starlette.datastructures import FormData
from starlette.responses import Response

from music_wrapped_api import _test_hooks

from ._decoders import (
    AppleStoreInput,
    decode_apple_store,
    decode_youtube_credentials,
)
from ._payloads import build_payload_for_service


async def _generate(request: Request) -> dict[str, str]:
    body = await request.body()
    doc = load_json_bytes(body)

    redis_url = _test_hooks.require_env("REDIS_URL")
    conn = _test_hooks.rq_conn(redis_url)
    queue = _test_hooks.rq_queue_factory(MUSIC_WRAPPED_QUEUE, conn)
    payload = build_payload_for_service(doc, redis_url=redis_url)
    job = queue.enqueue(
        "platform_music.jobs.process_wrapped_job",
        payload,
        job_timeout=600,
        result_ttl=86400,
        description=f"music_wrapped:{payload['year']}",
    )
    return {"job_id": job.get_id(), "status": "queued"}


def _strict_takeout_multipart(form: FormData) -> None:
    keys = set(form)
    if keys != {"file", "year"}:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="invalid multipart fields",
            http_status=400,
        )
    n_files = len(form.getlist("file"))
    n_years = len(form.getlist("year"))
    if n_files != 1 or n_years != 1:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="invalid multipart counts",
            http_status=400,
        )


async def _import_youtube_takeout(
    request: Request,
    file: Annotated[UploadFile, File(...)],
) -> dict[str, str]:
    form = await request.form()
    _strict_takeout_multipart(form)
    years = form.getlist("year")
    year_val = str(years[0])
    try:
        year = int(year_val)
    except ValueError:
        raise AppError(
            code=ErrorCode.INVALID_INPUT,
            message="year must be int",
            http_status=400,
        ) from None

    raw = await file.read()
    ctype = (file.content_type or "application/octet-stream").lower()

    from platform_core.json_utils import dump_json_str
    from platform_music.importers.youtube_takeout import parse_takeout_bytes

    plays = parse_takeout_bytes(raw, content_type=ctype)

    # Deterministic token for idempotency and cacheability
    token_id = hashlib.sha256(raw).hexdigest()[:32]

    redis_url = _test_hooks.require_env("REDIS_URL")
    redis = _test_hooks.redis_factory(redis_url)
    redis.set(f"ytmusic:takeout:{token_id}", dump_json_str(plays))

    # Enqueue import job
    conn = _test_hooks.rq_conn(redis_url)
    queue = _test_hooks.rq_queue_factory(MUSIC_WRAPPED_QUEUE, conn)
    payload: dict[str, JSONValue] = {
        "type": "music_wrapped.import_youtube_takeout.v1",
        "year": int(year),
        "token_id": token_id,
        "user_id": 0,
        "redis_url": redis_url,
        "queue_name": MUSIC_WRAPPED_QUEUE,
    }
    job = queue.enqueue(
        "platform_music.jobs.process_import_youtube_takeout",
        payload,
        job_timeout=600,
        result_ttl=86400,
        description=f"music_wrapped_import:{year}",
    )
    return {"job_id": job.get_id(), "status": "queued", "token_id": token_id}


async def _result(result_id: str) -> Response:
    redis_url = _test_hooks.require_env("REDIS_URL")
    redis = _test_hooks.redis_factory(redis_url)
    raw = redis.get(result_id)
    if raw is None:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="wrapped result not found",
            http_status=404,
        )
    return Response(content=raw, media_type="application/json")


async def _download(result_id: str) -> Response:
    redis_url = _test_hooks.require_env("REDIS_URL")
    redis = _test_hooks.redis_factory(redis_url)
    raw = redis.get(result_id)
    if raw is None:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="wrapped result not found",
            http_status=404,
        )
    from platform_core.json_utils import load_json_str
    from platform_music.wrapped import decode_wrapped_result

    doc = load_json_str(raw)
    result = decode_wrapped_result(doc)

    renderer = _test_hooks.build_renderer()
    png = renderer.render_wrapped(result)
    return Response(content=png, media_type="image/png")


def _build_lastfm_auth_url(callback: str, *, api_key: str) -> str:
    base = "https://www.last.fm/api/auth/"
    qs = urllib.parse.urlencode({"api_key": api_key, "cb": callback})
    return f"{base}?{qs}"


def _decode_lastfm_session(doc: dict[str, JSONValue]) -> tuple[str, str]:
    ses = doc.get("session")
    if not isinstance(ses, dict):
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message="missing session",
            http_status=502,
        )
    key_val = ses.get("key")
    name_val = ses.get("name")
    if not isinstance(key_val, str) or not isinstance(name_val, str):
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message="invalid session fields",
            http_status=502,
        )
    return key_val, name_val


async def _auth_lastfm_start(callback: str) -> dict[str, str]:
    api_key = _test_hooks.require_env("LASTFM_API_KEY")
    auth_url = _build_lastfm_auth_url(callback, api_key=api_key)
    return {"auth_url": auth_url}


async def _auth_lastfm_callback(token: str) -> dict[str, str]:
    api_key = _test_hooks.require_env("LASTFM_API_KEY")
    api_secret = _test_hooks.require_env("LASTFM_API_SECRET")
    doc = _test_hooks.lfm_get_session_json(api_key, api_secret, token)
    sk, name = _decode_lastfm_session(doc)
    return {"session_key": sk, "username": name}


async def _auth_spotify_start(callback: str) -> dict[str, str]:
    client_id = _test_hooks.require_env("SPOTIFY_CLIENT_ID")
    state = _test_hooks.rand_state()
    redis_url = _test_hooks.require_env("REDIS_URL")
    r = _test_hooks.redis_factory(redis_url)
    r.hset(f"spotify:state:{state}", {"ok": "1"})
    base = "https://accounts.spotify.com/authorize"
    params = {
        "client_id": client_id,
        "response_type": "code",
        "redirect_uri": callback,
        "scope": "user-read-recently-played user-top-read",
        "state": state,
    }
    url = base + "?" + urllib.parse.urlencode(params)
    return {"auth_url": url, "state": state}


async def _auth_spotify_callback(code: str, state: str, callback: str) -> dict[str, str | int]:
    redis_url = _test_hooks.require_env("REDIS_URL")
    r = _test_hooks.redis_factory(redis_url)
    st = r.hgetall(f"spotify:state:{state}")
    if st.get("ok") != "1":
        raise AppError(code=ErrorCode.INVALID_INPUT, message="invalid state", http_status=400)
    client_id = _test_hooks.require_env("SPOTIFY_CLIENT_ID")
    client_secret = _test_hooks.require_env("SPOTIFY_CLIENT_SECRET")
    doc = _test_hooks.spotify_exchange_code(code, callback, client_id, client_secret)
    at = doc.get("access_token")
    rt = doc.get("refresh_token")
    ex = doc.get("expires_in")
    if not isinstance(at, str) or not isinstance(rt, str) or not isinstance(ex, int):
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message="invalid token fields",
            http_status=502,
        )
    tok_id = hashlib.sha256(rt.encode("utf-8")).hexdigest()[:32]
    r.hset(
        f"spotify:session:{tok_id}",
        {"access_token": at, "refresh_token": rt, "expires_in": str(ex)},
    )
    return {"token_id": tok_id, "expires_in": ex}


async def _auth_youtube_store(request: Request) -> dict[str, str]:
    body = await request.body()
    from platform_core.json_utils import load_json_bytes

    doc = load_json_bytes(body)
    creds = decode_youtube_credentials(doc)
    # Store in Redis under a deterministic key
    token_id = hashlib.sha256(
        (creds["sapisid"] + ":" + creds["cookies"]).encode("utf-8")
    ).hexdigest()[:32]
    redis_url = _test_hooks.require_env("REDIS_URL")
    redis = _test_hooks.redis_factory(redis_url)
    redis.hset(
        f"ytmusic:session:{token_id}",
        {"sapisid": creds["sapisid"], "cookies": creds["cookies"]},
    )
    return {"token_id": token_id}


async def _auth_apple_store(request: Request) -> dict[str, str]:
    body = await request.body()
    from platform_core.json_utils import load_json_bytes

    doc = load_json_bytes(body)
    val: AppleStoreInput = decode_apple_store(doc)
    token_id = hashlib.sha256(val["music_user_token"].encode("utf-8")).hexdigest()[:32]
    redis_url = _test_hooks.require_env("REDIS_URL")
    redis = _test_hooks.redis_factory(redis_url)
    redis.hset(
        f"apple:session:{token_id}",
        {"music_user_token": val["music_user_token"]},
    )
    return {"token_id": token_id}


def _wrapped_result_schema() -> dict[str, JSONValue]:
    # Hand-authored JSON Schema matching platform_music.models.WrappedResult
    top_artist: dict[str, JSONValue] = {
        "type": "object",
        "properties": {
            "artist_name": {"type": "string"},
            "play_count": {"type": "integer", "minimum": 0},
        },
        "required": ["artist_name", "play_count"],
        "additionalProperties": False,
    }
    top_song: dict[str, JSONValue] = {
        "type": "object",
        "properties": {
            "title": {"type": "string"},
            "artist_name": {"type": "string"},
            "play_count": {"type": "integer", "minimum": 0},
        },
        "required": ["title", "artist_name", "play_count"],
        "additionalProperties": False,
    }
    by_month_entry: dict[str, JSONValue] = {
        "type": "object",
        "properties": {
            "month": {"type": "integer", "minimum": 1, "maximum": 12},
            "top_artists": {"type": "array", "items": top_artist},
        },
        "required": ["month", "top_artists"],
        "additionalProperties": False,
    }
    schema: dict[str, JSONValue] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "WrappedResult",
        "type": "object",
        "properties": {
            "service": {
                "type": "string",
                "enum": ["lastfm", "spotify", "apple_music", "youtube_music"],
            },
            "year": {"type": "integer"},
            "generated_at": {"type": "string"},
            "total_scrobbles": {"type": "integer", "minimum": 0},
            "top_artists": {"type": "array", "items": top_artist},
            "top_songs": {"type": "array", "items": top_song},
            "top_by_month": {"type": "array", "items": by_month_entry},
        },
        "required": [
            "service",
            "year",
            "generated_at",
            "total_scrobbles",
            "top_artists",
            "top_songs",
            "top_by_month",
        ],
        "additionalProperties": False,
    }
    return schema


async def _schema() -> Response:
    from platform_core.json_utils import dump_json_str

    return Response(
        content=dump_json_str(_wrapped_result_schema()),
        media_type="application/json",
    )


async def _status(job_id: str) -> Response:
    redis_url = _test_hooks.require_env("REDIS_URL")
    conn = _test_hooks.rq_conn(redis_url)
    job = _test_hooks.get_job(job_id, conn)
    status = job.get_status()
    meta = job.meta
    progress_val = meta.get("progress") if isinstance(meta, dict) else None
    progress = progress_val if isinstance(progress_val, int) else 0
    if job.is_finished and isinstance(job.result, str):
        rid: str | None = job.result
    else:
        rid = None
    from platform_core.json_utils import dump_json_str

    payload: dict[str, JSONValue] = {
        "job_id": job_id,
        "status": status,
        "progress": progress,
        "result_id": rid,
    }
    return Response(content=dump_json_str(payload), media_type="application/json")


_ROUTER = APIRouter(prefix="/v1/wrapped")
_ROUTER.add_api_route("/generate", _generate, methods=["POST"])
_ROUTER.add_api_route("/import/youtube-takeout", _import_youtube_takeout, methods=["POST"])
_ROUTER.add_api_route("/result/{result_id}", _result, methods=["GET"])
_ROUTER.add_api_route("/status/{job_id}", _status, methods=["GET"])
_ROUTER.add_api_route("/download/{result_id}", _download, methods=["GET"])
_ROUTER.add_api_route("/schema", _schema, methods=["GET"])
_ROUTER.add_api_route("/auth/lastfm/start", _auth_lastfm_start, methods=["GET"])
_ROUTER.add_api_route("/auth/lastfm/callback", _auth_lastfm_callback, methods=["GET"])
_ROUTER.add_api_route("/auth/spotify/start", _auth_spotify_start, methods=["GET"])
_ROUTER.add_api_route("/auth/spotify/callback", _auth_spotify_callback, methods=["GET"])
_ROUTER.add_api_route("/auth/youtube/store", _auth_youtube_store, methods=["POST"])
_ROUTER.add_api_route("/auth/apple/store", _auth_apple_store, methods=["POST"])


def build_router() -> APIRouter:
    return _ROUTER


__all__ = ["build_router"]
