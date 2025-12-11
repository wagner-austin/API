from __future__ import annotations

import io
import types
from pathlib import Path
from typing import Literal, Protocol

from model_trainer.api.routes import runs as runs_routes
from model_trainer.api.routes.runs import _BinaryFileProto
from model_trainer.core.config.settings import Settings
from model_trainer.core.services.container import ServiceContainer


class _SettingsFactory(Protocol):
    def __call__(
        self,
        *,
        artifacts_root: str | None = ...,
        runs_root: str | None = ...,
        logs_root: str | None = ...,
        data_root: str | None = ...,
        data_bank_api_url: str | None = ...,
        data_bank_api_key: str | None = ...,
        threads: int | None = ...,
        redis_url: str | None = ...,
        app_env: Literal["dev", "prod"] | None = ...,
        security_api_key: str | None = ...,
    ) -> Settings: ...


class _FakeBinaryFile:
    """Fake binary file that implements _BinaryFileProto for testing."""

    _buf: io.BytesIO

    def __init__(self, data: bytes) -> None:
        self._buf = io.BytesIO(data)

    def seek(self, offset: int, whence: int = 0) -> int:
        return self._buf.seek(offset, whence)

    def readline(self) -> bytes:
        return self._buf.readline()

    def __enter__(self) -> _FakeBinaryFile:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> bool | None:
        return None

    def __iter__(self) -> _FakeBinaryFile:
        return self

    def __next__(self) -> bytes:
        line = self._buf.readline()
        if not line:
            raise StopIteration
        return line


def test_runs_logs_stream_handles_oserror_and_ends(
    tmp_path: Path, settings_factory: _SettingsFactory
) -> None:
    # Arrange artifacts and a run log with two lines
    artifacts = tmp_path / "artifacts"
    run_id = "run-err"
    log_dir = artifacts / "models" / run_id
    log_dir.mkdir(parents=True)
    log_path = log_dir / "logs.jsonl"
    log_path.write_text("one\ntwo\n", encoding="utf-8")

    s = settings_factory(artifacts_root=str(artifacts))
    container = ServiceContainer.from_settings(s)
    h = runs_routes._RunsRoutes(container)

    # Track open calls
    calls: dict[str, int] = {"count": 0}

    def _fake_open(path: str, mode: str) -> _BinaryFileProto:
        if path.endswith("logs.jsonl") and "rb" in mode:
            calls["count"] += 1
            if calls["count"] >= 2:
                raise OSError("boom")
            data = Path(path).read_bytes()
            return _FakeBinaryFile(data)
        raise AssertionError("unexpected open mode or path in test stub")

    # Inject the fake open function via the class seam
    h._open_fn = _fake_open
    h._follow_max_loops = 3  # Ensure we don't loop forever

    # No sleep needed for this test
    def _no_sleep(_: float) -> None:
        pass

    h._sleep_fn = _no_sleep

    # Drive the iterator directly
    gen = h._sse_iter(str(log_path), tail=1, follow=True)
    out: list[bytes] = list(gen)

    # Assert: at least one SSE data line came through and stream ended quickly
    body = b"".join(out)
    assert b"data: " in body
    # And ensure our error path was triggered (second open attempted)
    assert calls["count"] >= 2
