from __future__ import annotations

from typing import Final, TypedDict

from platform_core.errors import AppError, ErrorCode
from platform_core.http_client import HttpxClient, build_client
from platform_core.json_utils import JSONValue, load_json_str
from platform_core.logging import get_logger

from ...config import DiscordbotSettings
from ...utils.youtube import extract_video_id, validate_youtube_url
from .api_client import TranscriptApiClient, captions


class TranscriptResult:
    __slots__ = ("text", "url", "video_id")

    def __init__(self, *, url: str, video_id: str, text: str) -> None:
        self.url = url
        self.video_id = video_id
        self.text = text

    def __getitem__(self, key: str) -> str:
        if key == "url":
            return self.url
        if key == "video_id":
            return self.video_id
        if key == "text":
            return self.text
        raise KeyError(key)


_DEFAULT_LANGS: Final[list[str]] = ["en", "en-US"]


def _parse_langs(raw: str) -> list[str]:
    parts = [p.strip() for p in raw.replace(",", " ").split()]
    langs = [p for p in parts if p]
    return langs if langs else list(_DEFAULT_LANGS)


class _TranscriptPayload(TypedDict, total=True):
    url: str
    video_id: str
    text: str


class TranscriptService:
    __slots__ = ("_client", "_logger", "_preferred_langs", "_timeout", "cfg")

    def __init__(self, cfg: DiscordbotSettings, client: HttpxClient | None = None) -> None:
        provider = cfg["transcript"]["provider"]
        base_url = cfg["transcript"]["api_url"]
        if provider != "api":
            raise RuntimeError("TranscriptService requires provider=api")
        if not base_url:
            raise RuntimeError("TRANSCRIPT_API_URL is required for transcript fetching")
        self.cfg = cfg
        self._client = client
        self._logger = get_logger(__name__)
        self._timeout = float(cfg["transcript"]["stt_api_timeout_seconds"])
        self._preferred_langs = _parse_langs(cfg["transcript"]["preferred_langs"])

    def _http_client(self) -> HttpxClient:
        if self._client is None:
            timeout_s = self._timeout
            self._client = build_client(timeout_s)
        return self._client

    def _client_dict(self) -> TranscriptApiClient:
        return {
            "base_url": self.cfg["transcript"]["api_url"],
            "timeout_seconds": self._timeout,
        }

    def _to_result(self, payload: _TranscriptPayload) -> TranscriptResult:
        return TranscriptResult(
            url=payload["url"],
            video_id=payload["video_id"],
            text=payload["text"],
        )

    def _validate_payload(self, payload: dict[str, JSONValue]) -> _TranscriptPayload:
        url_val: JSONValue = payload.get("url")
        vid_val: JSONValue = payload.get("video_id")
        text_val: JSONValue = payload.get("text")
        if isinstance(url_val, str) and isinstance(vid_val, str) and isinstance(text_val, str):
            return {"url": url_val, "video_id": vid_val, "text": text_val}
        raise RuntimeError("Transcript API payload missing required fields")

    def fetch_cleaned(self, url: str) -> TranscriptResult:
        canonical = validate_youtube_url(url)
        vid = extract_video_id(canonical)
        client_dict = self._client_dict()
        payload = captions(client_dict, url=canonical, preferred_langs=self._preferred_langs)
        validated = self._validate_payload(
            {"url": payload["url"], "video_id": payload["video_id"], "text": payload["text"]}
        )
        if validated["video_id"] != vid:
            self._logger.debug("Transcript video id mismatch; using extracted id")
            validated = {**validated, "video_id": vid}
        return self._to_result(validated)

    def fetch(self, url: str, request_id: str) -> TranscriptResult:
        base_url = self.cfg["transcript"]["api_url"]
        if not base_url:
            raise RuntimeError("TRANSCRIPT_API_URL is required for transcript fetching")
        resp = self._http_client().post(
            f"{base_url.rstrip('/')}/v1/transcript", headers={}, json={"url": url}
        )
        status = int(resp.status_code)
        if status == 200:
            data_raw = load_json_str(resp.text)
            if not isinstance(data_raw, dict):
                raise RuntimeError("Transcript API payload must be a JSON object")
            validated = self._validate_payload(data_raw)
            return self._to_result(validated)
        if status == 400:
            raise self._parse_app_error(resp.text, status)
        raise RuntimeError(f"Transcript API error: HTTP {status}")

    def _parse_app_error(self, text: str, status: int) -> AppError[ErrorCode]:
        parsed = load_json_str(text)
        if not isinstance(parsed, dict):
            raise RuntimeError("Transcript API error payload was not a JSON object")
        code_val = parsed.get("code")
        msg_val = parsed.get("message")
        if not isinstance(code_val, str) or not isinstance(msg_val, str):
            raise RuntimeError("Transcript API error payload missing code or message")
        code_map: dict[str, ErrorCode] = {c.value: c for c in ErrorCode}
        if code_val not in code_map:
            raise RuntimeError(f"Transcript API error code {code_val} is not recognized")
        return AppError(code_map[code_val], msg_val, http_status=status)


def fetch_cleaned(service: TranscriptService, url: str) -> TranscriptResult:
    """Module-level helper retained for tests; delegates to service."""
    return service.fetch_cleaned(url)


__all__ = ["TranscriptResult", "TranscriptService", "_parse_langs", "fetch_cleaned"]
