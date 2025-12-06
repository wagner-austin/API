from __future__ import annotations

from typing import TypedDict

from monorepo_guards._types import UnknownJson
from platform_core.errors import AppError, ErrorCode
from platform_core.http_client import HttpxClient, JsonObject, build_client
from platform_core.json_utils import JSONValue, load_json_str

HttpJsonPayload = JsonObject


class _CaptionsPayload(TypedDict, total=True):
    url: str
    preferred_langs: list[str] | None


class _SttPayload(TypedDict, total=True):
    url: str


class _TranscriptOut(TypedDict, total=True):
    url: str
    video_id: str
    text: str


class TranscriptApiClient(TypedDict):
    base_url: str
    timeout_seconds: float


def _post(client_dict: TranscriptApiClient, path: str, payload: HttpJsonPayload) -> _TranscriptOut:
    url = client_dict["base_url"].rstrip("/") + path
    client: HttpxClient = build_client(client_dict["timeout_seconds"])
    resp = client.post(url, json=payload, headers={})
    client.close()
    text = resp.text
    if resp.status_code == 400:
        stripped = text.lstrip()
        if stripped.startswith("{") and text.rstrip().endswith("}"):
            obj_any: UnknownJson = load_json_str(text)
            detail = obj_any.get("detail") if isinstance(obj_any, dict) else None
            if isinstance(detail, str):
                raise AppError(ErrorCode.INVALID_INPUT, detail, http_status=400)
        raise AppError(ErrorCode.INVALID_INPUT, "Invalid request", http_status=400)
    if resp.status_code != 200:
        raise RuntimeError(f"Transcript API error: {resp.status_code}")
    obj2: UnknownJson = load_json_str(text)
    if not isinstance(obj2, dict):
        raise RuntimeError("Unexpected API response format")
    url_val = obj2.get("url")
    vid_val = obj2.get("video_id")
    text_val = obj2.get("text")
    if isinstance(url_val, str) and isinstance(vid_val, str) and isinstance(text_val, str):
        url_out: str = url_val
        vid_out: str = vid_val
        text_out: str = text_val
        return {"url": url_out, "video_id": vid_out, "text": text_out}
    raise RuntimeError("Invalid transcript response payload")


def captions(
    client: TranscriptApiClient, *, url: str, preferred_langs: list[str] | None
) -> _TranscriptOut:
    langs: list[JSONValue] | None = list(preferred_langs) if preferred_langs is not None else None
    payload: HttpJsonPayload = {
        "url": url,
        "preferred_langs": langs,
    }
    out = _post(client, "/v1/captions", payload)
    return {"url": out["url"], "video_id": out["video_id"], "text": out["text"]}


def stt(client: TranscriptApiClient, *, url: str) -> _TranscriptOut:
    payload: HttpJsonPayload = {"url": url}
    out = _post(client, "/v1/stt", payload)
    return {"url": out["url"], "video_id": out["video_id"], "text": out["text"]}


__all__ = ["TranscriptApiClient", "captions", "stt"]
