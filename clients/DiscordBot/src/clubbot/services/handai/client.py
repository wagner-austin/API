from __future__ import annotations

from typing import Protocol

from platform_core.http_client import HttpxAsyncClient, HttpxResponse
from platform_core.http_utils import add_correlation_header
from platform_core.json_utils import JSONValue

from clubbot import _test_hooks


class PredictResult:
    __slots__ = ("confidence", "digit", "latency_ms", "model_id", "probs", "uncertain")

    def __init__(
        self,
        digit: int,
        confidence: float,
        probs: tuple[float, ...],
        model_id: str,
        uncertain: bool,
        latency_ms: int,
    ) -> None:
        self.digit = digit
        self.confidence = confidence
        self.probs = probs
        self.model_id = model_id
        self.uncertain = uncertain
        self.latency_ms = latency_ms

    def __getitem__(self, key: str) -> int | float | bool | str | tuple[float, ...]:
        if key == "digit":
            return self.digit
        if key == "confidence":
            return self.confidence
        if key == "probs":
            return self.probs
        if key == "model_id":
            return self.model_id
        if key == "uncertain":
            return self.uncertain
        if key == "latency_ms":
            return self.latency_ms
        raise KeyError(key)


class HandwritingAPIError(Exception):
    def __init__(
        self,
        status: int,
        message: str,
        *,
        code: str | None = None,
        request_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status = int(status)
        self.code = code
        self.request_id = request_id


class HandwritingReader(Protocol):
    async def read_digit(
        self,
        *,
        data: bytes,
        filename: str,
        content_type: str,
        request_id: str,
        center: bool,
        visualize: bool,
    ) -> PredictResult: ...


def _top_k_indices(probs: list[float] | tuple[float, ...], k: int = 3) -> list[int]:
    xs: list[tuple[int, float]] = [(i, float(p)) for i, p in enumerate(probs)]

    def _second(pair: tuple[int, float]) -> float:
        return pair[1]

    xs.sort(key=_second, reverse=True)
    return [xs[i][0] for i in range(min(k, len(xs)))]


class HandwritingClient(HandwritingReader):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: int = 5,
        max_retries: int = 1,
        client: HttpxAsyncClient | None = None,
    ) -> None:
        self._base: str = base_url.rstrip("/")
        self._api_key: str | None = (api_key or "").strip() or None
        self._timeout: float = float(timeout_seconds)
        self._retries: int = max(0, int(max_retries))
        self._client: HttpxAsyncClient = (
            _test_hooks.build_async_client(self._timeout) if client is None else client
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def read_digit(
        self,
        *,
        data: bytes,
        filename: str,
        content_type: str,
        request_id: str,
        center: bool = True,
        visualize: bool = False,
    ) -> PredictResult:
        url = (
            f"{self._base}/v1/read?center={'true' if center else 'false'}"
            f"&visualize={'true' if visualize else 'false'}"
        )
        headers: dict[str, str] = add_correlation_header({"Accept": "application/json"}, request_id)
        if self._api_key:
            headers["X-Api-Key"] = self._api_key
        files: dict[str, tuple[str, bytes, str]] = {"file": (filename, data, content_type)}
        return await self._attempt_read(url, headers, files)

    async def _attempt_read(
        self,
        url: str,
        headers: dict[str, str],
        files: dict[str, tuple[str, bytes, str]],
    ) -> PredictResult:
        resp = await self._client.post(url, headers=headers, files=files)
        if resp.status_code >= 400:
            raise _shape_api_error(resp)
        body_obj: JSONValue = resp.json()
        if not isinstance(body_obj, dict):
            raise HandwritingAPIError(500, "Invalid response body")
        return _decode_predict_result(body_obj)


def _decode_predict_result(d: dict[str, JSONValue]) -> PredictResult:
    def _decode_num(x: JSONValue) -> float:
        return float(str(x))

    digit = int(str(d.get("digit", 0)))
    confidence = _decode_num(d.get("confidence", 0.0))
    probs_val = d.get("probs", [])
    probs: tuple[float, ...] = (
        tuple(_decode_num(p) for p in probs_val) if isinstance(probs_val, list) else ()
    )
    model_id = str(d.get("model_id", ""))
    uncertain = bool(d.get("uncertain", False))
    latency_ms = int(str(d.get("latency_ms", 0)))
    return PredictResult(
        digit=digit,
        confidence=confidence,
        probs=probs,
        model_id=model_id,
        uncertain=uncertain,
        latency_ms=latency_ms,
    )


def _shape_api_error(resp: HttpxResponse) -> HandwritingAPIError:
    status = int(resp.status_code)
    code: str | None = None
    message = f"HTTP {status}"
    request_id: str | None = resp.headers.get("X-Request-ID")
    # Attempt to extract structured fields only if JSON content appears valid.
    # Avoid try/except to comply with strict guard rules.
    text = resp.text
    stripped = text.lstrip()
    if stripped.startswith("{") and text.rstrip().endswith("}"):
        obj: JSONValue = resp.json()
        if isinstance(obj, dict):
            raw_code = obj.get("code")
            code = str(raw_code) if isinstance(raw_code, str) else code
            raw_msg = obj.get("message")
            message = str(raw_msg) if isinstance(raw_msg, str) else message
            rid = obj.get("request_id")
            request_id = str(rid) if isinstance(rid, str) else request_id
    return HandwritingAPIError(status=status, message=message, code=code, request_id=request_id)
