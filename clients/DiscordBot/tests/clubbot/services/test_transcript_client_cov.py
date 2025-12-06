from __future__ import annotations

from collections.abc import Mapping

import pytest
from platform_core.errors import AppError
from platform_core.http_client import HttpxClient, HttpxResponse, SyncTransport
from platform_core.json_utils import InvalidJsonError, JSONValue, load_json_str
from tests.support.settings import build_settings

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


def test_client_dict_and_http_client_build(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    svc = TranscriptService(cfg)
    # Stub build_client to return a fake; the client is cached on first call
    resp0: HttpxResponse = _FakeResp(500, "")
    fake = _FakeClient(resp0)

    def _build_client(
        timeout_seconds: float, transport: SyncTransport | None = None
    ) -> HttpxClient:
        assert timeout_seconds == float(cfg["transcript"]["stt_api_timeout_seconds"])
        _ = transport
        return fake

    monkeypatch.setattr(
        "clubbot.services.transcript.client.build_client", _build_client, raising=True
    )
    # First call constructs and caches
    client = svc._http_client()
    assert type(client) is _FakeClient
    # Client dict reflects base URL and timeout
    d = svc._client_dict()
    assert isinstance(d["base_url"], str) and d["timeout_seconds"] == float(
        cfg["transcript"]["stt_api_timeout_seconds"]
    )


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


def test_parse_app_error_payload_validation(monkeypatch: pytest.MonkeyPatch) -> None:
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


def test_module_level_fetch_cleaned(monkeypatch: pytest.MonkeyPatch) -> None:
    from clubbot.services.transcript.client import fetch_cleaned as module_fetch

    cfg = _cfg()
    svc = TranscriptService(cfg)

    def _validate(u: str) -> str:
        return u

    def _extract(_u: str) -> str:
        return "vid"

    def _captions(
        _client: dict[str, JSONValue], *, url: str, preferred_langs: list[str]
    ) -> dict[str, str]:
        _ = (url, preferred_langs)
        return {"url": "u", "video_id": "vid", "text": "t"}

    monkeypatch.setattr(
        "clubbot.services.transcript.client.validate_youtube_url", _validate, raising=True
    )
    monkeypatch.setattr(
        "clubbot.services.transcript.client.extract_video_id", _extract, raising=True
    )
    monkeypatch.setattr("clubbot.services.transcript.client.captions", _captions, raising=True)
    res = module_fetch(svc, "http://y")
    assert type(res) is TranscriptResult


def test_fetch_cleaned_vid_mismatch_logs(monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _cfg()
    svc = TranscriptService(cfg)

    def _validate(u: str) -> str:
        return u

    def _extract(_u: str) -> str:
        return "different"

    def _captions(
        _client: dict[str, JSONValue], *, url: str, preferred_langs: list[str]
    ) -> dict[str, str]:
        _ = (url, preferred_langs)
        return {"url": "u", "video_id": "vid", "text": "t"}

    monkeypatch.setattr(
        "clubbot.services.transcript.client.validate_youtube_url", _validate, raising=True
    )
    monkeypatch.setattr(
        "clubbot.services.transcript.client.extract_video_id", _extract, raising=True
    )
    monkeypatch.setattr("clubbot.services.transcript.client.captions", _captions, raising=True)
    res = svc.fetch_cleaned("http://y")
    assert isinstance(res, TranscriptResult) and res.video_id == "different"
