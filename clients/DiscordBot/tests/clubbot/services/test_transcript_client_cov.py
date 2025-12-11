from __future__ import annotations

from collections.abc import Mapping

import pytest
from platform_core.errors import AppError
from platform_core.http_client import HttpxClient, HttpxResponse
from platform_core.json_utils import InvalidJsonError, JSONValue, load_json_str
from tests.support.settings import build_settings

from clubbot import _test_hooks
from clubbot.config import DiscordbotSettings
from clubbot.services.transcript.client import TranscriptResult, TranscriptService, _parse_langs


class _FakeResp:
    def __init__(self, status: int, text: str) -> None:
        self.status_code: int = status
        self.text: str = text
        self.headers: Mapping[str, str] = {}
        self.content: bytes | bytearray = b""

    def json(self) -> JSONValue:
        return load_json_str(self.text)


class _FakeClient:
    def __init__(self, resp: HttpxResponse) -> None:
        self._resp = resp
        self.calls: list[tuple[str, Mapping[str, str], JSONValue | None]] = []

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        json: JSONValue | None = None,
        files: Mapping[str, tuple[str, bytes, str]] | None = None,
    ) -> HttpxResponse:
        self.calls.append((url, headers, json))
        return self._resp

    def close(self) -> None:
        return None


def _cfg() -> DiscordbotSettings:
    return build_settings(transcript_provider="api")


def test_parse_langs_variants() -> None:
    assert _parse_langs("en, en-US") == ["en", "en-US"]
    assert _parse_langs("  ") == ["en", "en-US"]


def test_client_dict_and_http_client_build() -> None:
    cfg = _cfg()
    svc = TranscriptService(cfg)
    # Stub build_client to return a fake; the client is cached on first call
    resp0: HttpxResponse = _FakeResp(500, "")
    fake = _FakeClient(resp0)

    def _build_client(timeout: float) -> HttpxClient:
        assert timeout == float(cfg["transcript"]["stt_api_timeout_seconds"])
        return fake

    original = _test_hooks.build_client
    _test_hooks.build_client = _build_client
    try:
        # First call constructs and caches
        client = svc._http_client()
        assert type(client) is _FakeClient
        # Client dict reflects base URL and timeout
        d = svc._client_dict()
        assert isinstance(d["base_url"], str) and d["timeout_seconds"] == float(
            cfg["transcript"]["stt_api_timeout_seconds"]
        )
    finally:
        _test_hooks.build_client = original


def test_fetch_success_and_payload_validation() -> None:
    cfg = _cfg()
    body = {"url": "http://x", "video_id": "vid", "text": "t"}
    payload = "{" + ",".join(f'"{k}": "{v}"' for k, v in body.items()) + "}"
    resp: HttpxResponse = _FakeResp(200, payload)
    fake = _FakeClient(resp)
    svc = TranscriptService(cfg, client=fake)
    res = svc.fetch("http://x", request_id="r")
    assert type(res) is TranscriptResult
    assert res.text == "t"


def test_fetch_400_app_error() -> None:
    cfg = _cfg()
    err = {"code": "INVALID_INPUT", "message": "bad"}
    payload = "{" + ",".join(f'"{k}": "{v}"' for k, v in err.items()) + "}"
    resp1: HttpxResponse = _FakeResp(400, payload)
    fake = _FakeClient(resp1)
    svc = TranscriptService(cfg, client=fake)
    with pytest.raises(AppError) as excinfo:
        _ = svc.fetch("http://x", request_id="r")
    assert type(excinfo.value) is AppError


def test_fetch_non_json_payload_raises() -> None:
    cfg = _cfg()
    resp2: HttpxResponse = _FakeResp(200, "not-json")
    fake = _FakeClient(resp2)
    svc = TranscriptService(cfg, client=fake)
    with pytest.raises(InvalidJsonError):
        _ = svc.fetch("http://x", request_id="r")


def test_fetch_unexpected_status_raises() -> None:
    cfg = _cfg()
    resp3: HttpxResponse = _FakeResp(503, "")
    fake = _FakeClient(resp3)
    svc = TranscriptService(cfg, client=fake)
    with pytest.raises(RuntimeError):
        _ = svc.fetch("http://x", request_id="r")


def test_parse_app_error_payload_validation() -> None:
    cfg = _cfg()
    svc = TranscriptService(cfg)
    with pytest.raises(InvalidJsonError):
        _ = svc._parse_app_error("not-json", 400)
    with pytest.raises(RuntimeError):
        _ = svc._parse_app_error("{}", 400)
    with pytest.raises(RuntimeError):
        _ = svc._parse_app_error('{"code": "x", "message": "m"}', 400)


def test_validate_payload_missing_fields_raises() -> None:
    cfg = _cfg()
    svc = TranscriptService(cfg)
    with pytest.raises(RuntimeError):
        _ = svc._validate_payload({"url": "u"})


def test_fetch_missing_base_url_raises() -> None:
    cfg = _cfg()
    svc = TranscriptService(cfg)
    # Invalidate base URL after init
    cfg["transcript"]["api_url"] = ""
    with pytest.raises(RuntimeError):
        _ = svc.fetch("http://x", request_id="r")


def test_fetch_200_non_dict_payload_raises() -> None:
    cfg = _cfg()
    resp: HttpxResponse = _FakeResp(200, "[]")
    fake = _FakeClient(resp)
    svc = TranscriptService(cfg, client=fake)
    with pytest.raises(RuntimeError):
        _ = svc.fetch("http://x", request_id="r")


def test_parse_app_error_parsed_not_dict() -> None:
    cfg = _cfg()
    svc = TranscriptService(cfg)
    with pytest.raises(InvalidJsonError):
        _ = svc._parse_app_error("not-json", 400)
    with pytest.raises(RuntimeError):
        _ = svc._parse_app_error("[]", 400)


def test_module_level_fetch_cleaned() -> None:
    from clubbot.services.transcript.client import fetch_cleaned as module_fetch

    cfg = _cfg()
    svc = TranscriptService(cfg)

    def _validate(url: str) -> str:
        return url

    def _extract(url: str) -> str:
        _ = url
        return "vid"

    def _captions(
        client: dict[str, float | str], *, url: str, preferred_langs: list[str]
    ) -> dict[str, str]:
        _ = (client, url, preferred_langs)
        return {"url": "u", "video_id": "vid", "text": "t"}

    original_validate = _test_hooks.validate_youtube_url_for_client
    original_extract = _test_hooks.extract_video_id
    original_captions = _test_hooks.captions
    _test_hooks.validate_youtube_url_for_client = _validate
    _test_hooks.extract_video_id = _extract
    _test_hooks.captions = _captions
    try:
        res = module_fetch(svc, "http://y")
        assert type(res) is TranscriptResult
    finally:
        _test_hooks.validate_youtube_url_for_client = original_validate
        _test_hooks.extract_video_id = original_extract
        _test_hooks.captions = original_captions


def test_fetch_cleaned_vid_mismatch_logs() -> None:
    cfg = _cfg()
    svc = TranscriptService(cfg)

    def _validate(url: str) -> str:
        return url

    def _extract(url: str) -> str:
        _ = url
        return "different"

    def _captions(
        client: dict[str, float | str], *, url: str, preferred_langs: list[str]
    ) -> dict[str, str]:
        _ = (client, url, preferred_langs)
        return {"url": "u", "video_id": "vid", "text": "t"}

    original_validate = _test_hooks.validate_youtube_url_for_client
    original_extract = _test_hooks.extract_video_id
    original_captions = _test_hooks.captions
    _test_hooks.validate_youtube_url_for_client = _validate
    _test_hooks.extract_video_id = _extract
    _test_hooks.captions = _captions
    try:
        res = svc.fetch_cleaned("http://y")
        assert isinstance(res, TranscriptResult) and res.video_id == "different"
    finally:
        _test_hooks.validate_youtube_url_for_client = original_validate
        _test_hooks.extract_video_id = original_extract
        _test_hooks.captions = original_captions
