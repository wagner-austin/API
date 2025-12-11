from __future__ import annotations

from typing import Protocol, TypedDict

from platform_core.errors import AppError, ErrorCode
from platform_core.http_client import HttpxClient, JsonObject
from platform_core.http_utils import add_correlation_header
from platform_core.json_utils import load_json_str
from platform_core.logging import get_logger

from clubbot import _test_hooks

from ...config import DiscordbotSettings


class QRRequestPayload(TypedDict, total=False):
    url: str
    ecc: str
    box_size: int
    border: int
    fill_color: str
    back_color: str


class QRResult:
    __slots__ = ("image_png", "url")

    def __init__(self, *, image_png: bytes, url: str) -> None:
        self.image_png = image_png
        self.url = url

    def __getitem__(self, key: str) -> bytes | str:
        if key == "image_png":
            return self.image_png
        if key == "url":
            return self.url
        raise KeyError(key)


class QRClient(Protocol):
    def png(self, *, payload: QRRequestPayload, request_id: str) -> bytes: ...


class QRHttpClient(QRClient):
    def __init__(self, base_url: str, timeout_seconds: int = 5) -> None:
        self._base: str = base_url.rstrip("/")
        self._timeout: float = float(timeout_seconds)
        self._client: HttpxClient = _test_hooks.build_client(self._timeout)

    def close(self) -> None:
        self._client.close()

    def png(
        self,
        *,
        payload: QRRequestPayload,
        request_id: str,
    ) -> bytes:
        url_value = payload.get("url") if isinstance(payload, dict) else None
        if not isinstance(url_value, str):
            raise ValueError("QR request payload must include url")
        endpoint = f"{self._base}/v1/qr"
        headers = add_correlation_header({"Accept": "image/png"}, request_id)
        body = self._build_body(payload)
        resp = self._client.post(endpoint, headers=headers, json=body)
        status = resp.status_code
        if status == 200:
            return self._parse_png_response(resp.content)
        if status == 400:
            raise self._decode_app_error(resp.text, status)
        raise RuntimeError(f"QR API error: HTTP {status}")

    @staticmethod
    def _parse_png_response(body_bytes: bytes | bytearray) -> bytes:
        png_header = b"\x89PNG\r\n\x1a\n"
        is_bytes = isinstance(body_bytes, (bytes | bytearray))
        if not (is_bytes and bytes(body_bytes)[:8] == png_header):
            raise RuntimeError("QR API did not return a valid PNG image")
        return bytes(body_bytes)

    @staticmethod
    def _decode_app_error(text: str, status: int) -> AppError[ErrorCode]:
        parsed_raw = load_json_str(text)
        if not isinstance(parsed_raw, dict):
            raise RuntimeError("QR API error payload was not a JSON object")
        code_val = parsed_raw.get("code")
        msg_val = parsed_raw.get("message")
        if not isinstance(code_val, str) or not isinstance(msg_val, str):
            raise RuntimeError("QR API error payload missing code or message")
        code_map: dict[str, ErrorCode] = {code.value: code for code in ErrorCode}
        if code_val not in code_map:
            raise RuntimeError(f"QR API error code {code_val} is not recognized")
        return AppError(code_map[code_val], msg_val, http_status=status)

    @staticmethod
    def _build_body(payload: QRRequestPayload) -> JsonObject:
        body: JsonObject = {}
        _QRHttpClientHelpers.copy_str(payload, body, "url")
        _QRHttpClientHelpers.copy_str(payload, body, "ecc")
        _QRHttpClientHelpers.copy_int(payload, body, "box_size")
        _QRHttpClientHelpers.copy_int(payload, body, "border")
        _QRHttpClientHelpers.copy_str(payload, body, "fill_color")
        _QRHttpClientHelpers.copy_str(payload, body, "back_color")
        return body


class _QRHttpClientHelpers:
    @staticmethod
    def copy_str(source: QRRequestPayload, target: JsonObject, key: str) -> None:
        if key in source:
            value = source.get(key)
            if isinstance(value, str):
                target[key] = value

    @staticmethod
    def copy_int(source: QRRequestPayload, target: JsonObject, key: str) -> None:
        if key in source:
            value = source.get(key)
            if isinstance(value, int):
                target[key] = value


class QRService:
    __slots__ = ("_client", "_logger", "cfg")

    def __init__(self, cfg: DiscordbotSettings, client: QRClient | None = None) -> None:
        self.cfg = cfg
        self._client = client
        self._logger = get_logger(__name__)

    def _get_client(self) -> QRClient:
        if self._client is not None:
            return self._client
        base = self.cfg["qr"]["api_url"]
        if not base:
            raise RuntimeError("QR_API_URL is required for QR generation")
        timeout = 5
        self._client = QRHttpClient(base, timeout_seconds=timeout)
        return self._client

    def generate_qr(self, url: str) -> QRResult:
        return self.generate_qr_with_payload({"url": url})

    def generate_qr_with_payload(self, payload: QRRequestPayload) -> QRResult:
        url_value = payload.get("url") if isinstance(payload, dict) else None
        if not isinstance(url_value, str):
            raise ValueError("QR request payload must include url")
        self._logger.debug(
            "QRService generating via API for url=%s",
            url_value,
        )
        png = self._get_client().png(
            payload=payload,
            request_id="qr",
        )
        return QRResult(image_png=png, url=url_value)


__all__ = ["QRClient", "QRHttpClient", "QRRequestPayload", "QRResult", "QRService"]
