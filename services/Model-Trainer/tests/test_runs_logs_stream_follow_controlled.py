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
    """Fake binary file wrapping BytesIO that implements _BinaryFileProto."""

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


class _Reader:
    """Controlled reader that emits data then returns empty on subsequent reads."""

    _emitted_empty: bool
    _data: list[bytes]

    def __init__(self, data: bytes) -> None:
        self._emitted_empty = False
        self._data = [*data.splitlines(keepends=True)]

    def __enter__(self) -> _Reader:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> bool | None:
        return None

    def seek(self, offset: int, whence: int = 0) -> int:
        return 0

    def readline(self) -> bytes:
        if not self._emitted_empty:
            self._emitted_empty = True
            return b""
        return self._data.pop(0) if self._data else b""

    def __iter__(self) -> _Reader:
        return self

    def __next__(self) -> bytes:
        line = self.readline()
        if not line:
            raise StopIteration
        return line


def test_runs_logs_stream_follow_else_branch_exits_quickly(
    tmp_path: Path,
    settings_factory: _SettingsFactory,
) -> None:
    # Arrange artifacts and a single-line log file
    artifacts = tmp_path / "artifacts"
    run_id = "run-follow"
    run_dir = artifacts / "models" / run_id
    run_dir.mkdir(parents=True)
    (run_dir / "logs.jsonl").write_text("one\n", encoding="utf-8")

    s = settings_factory(artifacts_root=str(artifacts))
    container = ServiceContainer.from_settings(s)
    h = runs_routes._RunsRoutes(container)

    # Inject deterministic seams: no sleep, early-EOF reader, and finite follow loops
    sleep_calls = {"n": 0}

    def _no_sleep(_: float) -> None:
        sleep_calls["n"] += 1
        return

    h._sleep_fn = _no_sleep
    h._follow_max_loops = 1

    # Create a fake open that returns a controlled reader for the follow phase
    call_state: dict[str, int] = {"n": 0}

    def _fake_open(path: str, mode: str) -> _BinaryFileProto:
        call_state["n"] += 1
        if call_state["n"] == 1:
            # First call: read the real file for tail
            return _FakeBinaryFile(Path(path).read_bytes())
        if path.endswith("logs.jsonl") and "rb" in mode:
            # Second call: follow phase - use controlled reader
            return _Reader(b"two\n")
        raise AssertionError("unexpected open call")

    h._open_fn = _fake_open

    # Drive the iterator directly without HTTP to avoid timeouts
    gen = h._sse_iter(str(run_dir / "logs.jsonl"), tail=1, follow=True)
    out: list[bytes] = list(gen)
    # Assert: initial tail emitted, sleep branch executed once, then closed
    assert len(out) >= 1 and out[0].startswith(b"data: ")
    assert sleep_calls["n"] >= 1
