from __future__ import annotations

import hashlib
import urllib.parse
import urllib.request
from types import TracebackType
from typing import Annotated, Literal, Protocol, TypedDict

from fastapi import APIRouter, File, Request, UploadFile
from platform_core.config import _require_env_str
from platform_core.errors import AppError, ErrorCode
from platform_core.json_utils import JSONValue, load_json_bytes
from platform_core.queues import MUSIC_WRAPPED_QUEUE
from platform_music import WrappedResult
from platform_music.jobs import LastFmCredentials
from platform_workers.redis import redis_for_kv
from platform_workers.rq_harness import RQClientQueue, _RedisBytesClient, redis_raw_for_rq, rq_queue
from starlette.datastructures import FormData
from starlette.responses import Response

# Make __import__ overrideable in tests via monkeypatch
__import__ = __import__

from ._decoders import (
    AppleGenerateFull,
    AppleGenerateToken,
    AppleStoreInput,
    LastFmGenerate,
    SpotifyGenerateFull,
    SpotifyGenerateToken,
    YouTubeGenerateFull,
    YouTubeGenerateToken,
    _LastFmCredsFull,
    _LastFmCredsSessionOnly,
    decode_apple_store,
    decode_wrapped_generate,
    decode_youtube_credentials,
    is_full_lastfm_credentials,
    to_full_lastfm_credentials,
)


class _GenerateRequest(TypedDict):
    year: int
    service: str


def _rq_conn(url: str) -> _RedisBytesClient:
    return redis_raw_for_rq(url)


def _rq_queue(name: str, *, connection: _RedisBytesClient) -> RQClientQueue:
    return rq_queue(name, connection=connection)


class _RQJobLike(Protocol):
    def get_status(self) -> str: ...

    @property
    def is_finished(self) -> bool: ...

    @property
    def meta(self) -> dict[str, JSONValue]: ...

    @property
    def result(self) -> str | None: ...


class _JobFetcher(Protocol):
    def __call__(self, job_id: str, *, connection: _RedisBytesClient) -> _RQJobLike: ...


class _RQJobClass(Protocol):
    def fetch(self, job_id: str, *, connection: _RedisBytesClient) -> _RQJobLike: ...


def _get_job(job_id: str, connection: _RedisBytesClient) -> _RQJobLike:
    """Load an RQ job instance using a typed dynamic import.

    Uses __import__("rq.job", fromlist=["Job"]) and a Protocol-typed class
    reference to avoid Any while calling the classmethod `fetch`.
    """
    rq_job_mod = __import__("rq.job", fromlist=["Job"])
    job_cls: _RQJobClass = rq_job_mod.Job
    return job_cls.fetch(job_id, connection=connection)


class _DecodeReq(TypedDict):
    year: int
    service: Literal["lastfm"]
    credentials: LastFmCredentials


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
            api_key_env=_require_env_str("LASTFM_API_KEY"),
            api_secret_env=_require_env_str("LASTFM_API_SECRET"),
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
    data = redis_for_kv(redis_url).hgetall(f"spotify:session:{req_sp['credentials']['token_id']}")
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
    data2 = redis_for_kv(redis_url).hgetall(f"apple:session:{req_ap['credentials']['token_id']}")
    mus = data2.get("music_user_token")
    if not isinstance(mus, str):
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="apple token not found",
            http_status=404,
        )
    creds_json: dict[str, JSONValue] = {
        "music_user_token": mus,
        "developer_token": _require_env_str("APPLE_DEVELOPER_TOKEN"),
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
    data3 = redis_for_kv(redis_url).hgetall(f"ytmusic:session:{req_yt['credentials']['token_id']}")
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


def _build_payload_for_service(doc: JSONValue, *, redis_url: str) -> dict[str, JSONValue]:
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


async def _generate(request: Request) -> dict[str, str]:
    body = await request.body()
    doc = load_json_bytes(body)

    redis_url = _require_env_str("REDIS_URL")
    conn = _rq_conn(redis_url)
    queue = _rq_queue(MUSIC_WRAPPED_QUEUE, connection=conn)
    payload = _build_payload_for_service(doc, redis_url=redis_url)
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

    redis_url = _require_env_str("REDIS_URL")
    redis = redis_for_kv(redis_url)
    redis.set(f"ytmusic:takeout:{token_id}", dump_json_str(plays))

    # Enqueue import job
    conn = _rq_conn(redis_url)
    queue = _rq_queue(MUSIC_WRAPPED_QUEUE, connection=conn)
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
    redis_url = _require_env_str("REDIS_URL")
    redis = redis_for_kv(redis_url)
    raw = redis.get(result_id)
    if raw is None:
        raise AppError(
            code=ErrorCode.NOT_FOUND,
            message="wrapped result not found",
            http_status=404,
        )
    return Response(content=raw, media_type="application/json")


class _RendererProto(Protocol):
    def render_wrapped(self, result: WrappedResult) -> bytes: ...


class _RendererFactory(Protocol):
    def __call__(self) -> _RendererProto: ...


async def _download(result_id: str) -> Response:
    redis_url = _require_env_str("REDIS_URL")
    redis = redis_for_kv(redis_url)
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

    img_mod = __import__("platform_music.image_gen.renderer", fromlist=["build_renderer"])
    factory: _RendererFactory = img_mod.build_renderer
    renderer = factory()
    png = renderer.render_wrapped(result)
    return Response(content=png, media_type="image/png")


def _build_lastfm_auth_url(callback: str, *, api_key: str) -> str:
    base = "https://www.last.fm/api/auth/"
    qs = urllib.parse.urlencode({"api_key": api_key, "cb": callback})
    return f"{base}?{qs}"


def _lastfm_sig(api_key: str, api_secret: str, token: str) -> str:
    raw = f"api_key{api_key}methodauth.getSessiontoken{token}{api_secret}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


class _HttpRespProto(Protocol):
    def read(self) -> bytes: ...

    def __enter__(self) -> _HttpRespProto: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class _UrlOpenModule(Protocol):
    def urlopen(self, url: str, timeout: float) -> _HttpRespProto: ...


def _lfm_get_session_json(api_key: str, api_secret: str, token: str) -> dict[str, JSONValue]:
    sig = _lastfm_sig(api_key, api_secret, token)
    params = {
        "method": "auth.getSession",
        "api_key": api_key,
        "api_sig": sig,
        "token": token,
        "format": "json",
    }
    url = "https://ws.audioscrobbler.com/2.0/?" + urllib.parse.urlencode(params)
    mod: _UrlOpenModule = __import__("urllib.request", fromlist=["urlopen"])
    with mod.urlopen(url, timeout=10) as resp:
        data = resp.read().decode("utf-8")
    from platform_core.json_utils import load_json_str

    doc = load_json_str(data)
    if not isinstance(doc, dict):
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message="invalid lastfm json",
            http_status=502,
        )
    return doc


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
    api_key = _require_env_str("LASTFM_API_KEY")
    auth_url = _build_lastfm_auth_url(callback, api_key=api_key)
    return {"auth_url": auth_url}


async def _auth_lastfm_callback(token: str) -> dict[str, str]:
    api_key = _require_env_str("LASTFM_API_KEY")
    api_secret = _require_env_str("LASTFM_API_SECRET")
    doc = _lfm_get_session_json(api_key, api_secret, token)
    sk, name = _decode_lastfm_session(doc)
    return {"session_key": sk, "username": name}


def _rand_state() -> str:
    import secrets as _secrets

    return _secrets.token_urlsafe(16)


async def _auth_spotify_start(callback: str) -> dict[str, str]:
    client_id = _require_env_str("SPOTIFY_CLIENT_ID")
    state = _rand_state()
    redis_url = _require_env_str("REDIS_URL")
    r = redis_for_kv(redis_url)
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


class _HttpPostRespProto(Protocol):
    def read(self) -> bytes: ...

    def __enter__(self) -> _HttpPostRespProto: ...

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None: ...


class _UrlOpenPostModule(Protocol):
    def urlopen(self, req: _RequestObj, timeout: float) -> _HttpPostRespProto: ...


class _RequestObj(Protocol):
    def add_header(self, name: str, value: str) -> None: ...


class _RequestFactory(Protocol):
    def __call__(self, url: str, data: bytes) -> _RequestObj: ...


def _spotify_exchange_code(
    code: str, redirect_uri: str, *, client_id: str, client_secret: str
) -> dict[str, JSONValue]:
    token_url = "https://accounts.spotify.com/api/token"
    data = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": redirect_uri,
    }
    encoded = urllib.parse.urlencode(data).encode("utf-8")

    class _Req:
        def __init__(self, url: str, data: bytes) -> None:
            self._ = (url, data)

        def add_header(self, name: str, value: str) -> None:
            _ = (name, value)

    req: _RequestObj = _Req(token_url, encoded)
    import base64 as _b64

    auth = (client_id + ":" + client_secret).encode("utf-8")
    hdr = "Basic " + _b64.b64encode(auth).decode("utf-8")
    req.add_header("Authorization", hdr)
    req.add_header("Content-Type", "application/x-www-form-urlencoded")
    imp = __import__
    mod: _UrlOpenPostModule = imp("urllib.request", fromlist=["urlopen"])
    with mod.urlopen(req, timeout=10) as resp:
        payload = resp.read().decode("utf-8")
    from platform_core.json_utils import load_json_str

    doc = load_json_str(payload)
    if not isinstance(doc, dict):
        raise AppError(
            code=ErrorCode.EXTERNAL_SERVICE_ERROR,
            message="invalid spotify json",
            http_status=502,
        )
    return doc


async def _auth_spotify_callback(code: str, state: str, callback: str) -> dict[str, str | int]:
    redis_url = _require_env_str("REDIS_URL")
    r = redis_for_kv(redis_url)
    st = r.hgetall(f"spotify:state:{state}")
    if st.get("ok") != "1":
        raise AppError(code=ErrorCode.INVALID_INPUT, message="invalid state", http_status=400)
    client_id = _require_env_str("SPOTIFY_CLIENT_ID")
    client_secret = _require_env_str("SPOTIFY_CLIENT_SECRET")
    doc = _spotify_exchange_code(code, callback, client_id=client_id, client_secret=client_secret)
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
    redis_url = _require_env_str("REDIS_URL")
    redis = redis_for_kv(redis_url)
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
    redis_url = _require_env_str("REDIS_URL")
    redis = redis_for_kv(redis_url)
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
    redis_url = _require_env_str("REDIS_URL")
    conn = _rq_conn(redis_url)
    job = _get_job(job_id, connection=conn)
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
