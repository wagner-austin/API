from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol

from platform_core.http_client import (
    HttpxAsyncClient,
    HttpxResponse,
    JsonObject,
    build_async_client,
)
from platform_core.http_utils import add_correlation_header
from platform_core.json_utils import InvalidJsonError, JSONValue, load_json_str
from platform_core.logging import get_logger

_logger = get_logger(__name__)


class TrainResponse:
    __slots__ = ("job_id", "run_id")

    def __init__(self, run_id: str, job_id: str) -> None:
        self.run_id = run_id
        self.job_id = job_id

    def __getitem__(self, key: str) -> str:
        if key == "run_id":
            return self.run_id
        if key == "job_id":
            return self.job_id
        raise KeyError(key)


class RunStatus:
    __slots__ = ("last_heartbeat_ts", "message", "run_id", "status")

    def __init__(
        self,
        *,
        run_id: str,
        status: str,
        last_heartbeat_ts: float | None,
        message: str | None,
    ) -> None:
        self.run_id = run_id
        self.status = status
        self.last_heartbeat_ts = last_heartbeat_ts
        self.message = message

    def __getitem__(self, key: str) -> str | float | None:
        if key == "run_id":
            return self.run_id
        if key == "status":
            return self.status
        if key == "last_heartbeat_ts":
            return self.last_heartbeat_ts
        if key == "message":
            return self.message
        raise KeyError(key)


class ModelTrainerAPIError(Exception):
    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = int(status)


class ModelTrainerClient(Protocol):
    async def train(
        self,
        *,
        user_id: int,
        model_family: str,
        model_size: str,
        max_seq_len: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        corpus_path: str,
        tokenizer_id: str,
        request_id: str,
    ) -> TrainResponse: ...

    async def status(self, *, run_id: str, request_id: str) -> RunStatus: ...


class HTTPModelTrainerClient(ModelTrainerClient):
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_seconds: int = 10,
        client: HttpxAsyncClient | None = None,
    ) -> None:
        self._base = base_url.rstrip("/")
        api_key_trimmed = (api_key or "").strip()
        self._api_key = api_key_trimmed if api_key_trimmed != "" else None
        self._client: HttpxAsyncClient = (
            build_async_client(float(timeout_seconds)) if client is None else client
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def train(
        self,
        *,
        user_id: int,
        model_family: str,
        model_size: str,
        max_seq_len: int,
        num_epochs: int,
        batch_size: int,
        learning_rate: float,
        corpus_path: str,
        tokenizer_id: str,
        request_id: str,
    ) -> TrainResponse:
        url = f"{self._base}/runs/train"
        headers = self._headers(request_id)
        body: JsonObject = {
            "model_family": model_family,
            "model_size": model_size,
            "max_seq_len": int(max_seq_len),
            "num_epochs": int(num_epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "corpus_path": corpus_path,
            "tokenizer_id": tokenizer_id,
            "user_id": int(user_id),
        }
        resp = await self._client.post(url, headers=headers, json=body)
        if resp.status_code >= 400:
            raise ModelTrainerAPIError(int(resp.status_code), _extract_message(resp))
        payload = _load_json_dict(resp.text)
        return _parse_train_response(payload)

    async def status(self, *, run_id: str, request_id: str) -> RunStatus:
        url = f"{self._base}/runs/{run_id}"
        headers = self._headers(request_id)
        resp = await self._client.get(url, headers=headers)
        if resp.status_code >= 400:
            raise ModelTrainerAPIError(int(resp.status_code), _extract_message(resp))
        payload = _load_json_dict(resp.text)
        return _parse_status_response(payload)

    def _headers(self, request_id: str) -> dict[str, str]:
        headers: dict[str, str] = {"Accept": "application/json"}
        headers = add_correlation_header(headers, request_id)
        api_key = self._api_key
        if api_key is not None:
            headers["X-Api-Key"] = api_key
        return headers


def _load_json_dict(raw_text: str) -> dict[str, JSONValue]:
    try:
        parsed = load_json_str(raw_text)
    except InvalidJsonError as exc:
        raise ModelTrainerAPIError(500, "Invalid response body") from exc
    if isinstance(parsed, dict):
        return parsed
    raise ModelTrainerAPIError(500, "Invalid response body")


def _require_str_field(obj: Mapping[str, JSONValue], field: str) -> str:
    value = obj.get(field)
    if isinstance(value, str):
        return value
    raise ModelTrainerAPIError(500, "Invalid response body")


def _require_optional_str(obj: Mapping[str, JSONValue], field: str) -> str | None:
    value = obj.get(field)
    if value is None:
        return None
    if isinstance(value, str):
        return value
    raise ModelTrainerAPIError(500, "Invalid response body")


def _parse_last_heartbeat(value: JSONValue) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError as exc:
            raise ModelTrainerAPIError(500, "Invalid response body") from exc
    raise ModelTrainerAPIError(500, "Invalid response body")


def _parse_train_response(obj: Mapping[str, JSONValue]) -> TrainResponse:
    run_id = _require_str_field(obj, "run_id")
    job_id = _require_str_field(obj, "job_id")
    return TrainResponse(run_id, job_id)


def _parse_status_response(obj: Mapping[str, JSONValue]) -> RunStatus:
    run_id = _require_str_field(obj, "run_id")
    status = _require_str_field(obj, "status")
    last = _parse_last_heartbeat(obj.get("last_heartbeat_ts"))
    msg = _require_optional_str(obj, "message")
    return RunStatus(run_id=run_id, status=status, last_heartbeat_ts=last, message=msg)


def _extract_message(resp: HttpxResponse) -> str:
    text = resp.text
    try:
        parsed = load_json_str(text)
    except InvalidJsonError:
        _logger.debug("model_trainer_client: response body not JSON; falling back to HTTP status")
        return f"HTTP {int(resp.status_code)}"
    if isinstance(parsed, dict):
        msg_val = parsed.get("message") or parsed.get("detail") or parsed.get("error")
        if isinstance(msg_val, str) and msg_val.strip() != "":
            return msg_val
    return f"HTTP {int(resp.status_code)}"


__all__ = [
    "HTTPModelTrainerClient",
    "ModelTrainerAPIError",
    "ModelTrainerClient",
    "RunStatus",
    "TrainResponse",
    "build_async_client",
]
