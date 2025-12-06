from __future__ import annotations

import io
import types
from collections.abc import Generator
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Literal, Protocol

import pytest

from model_trainer.api.routes import runs as runs_routes
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


class _Reader:
    def __init__(self: _Reader, data: bytes) -> None:
        self._emitted_empty: bool = False
        self._data: list[bytes] = [*data.splitlines(keepends=True)]

    def __enter__(self: _Reader) -> _Reader:
        return self

    def __exit__(
        self: _Reader,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: types.TracebackType | None,
    ) -> Literal[False]:
        return False

    def seek(self: _Reader, offset: int, whence: int = 0) -> int:
        # Seek not used meaningfully in the stub for this test
        return 0

    def readline(self: _Reader) -> bytes:
        if not self._emitted_empty:
            self._emitted_empty = True
            return b""
        return self._data.pop(0) if self._data else b""


def test_runs_logs_stream_follow_else_branch_exits_quickly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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

    # Patch module-level open so the first call reads real tail data,
    # and the second (follow) uses our controlled reader

    call_state = {"n": 0}

    from contextlib import contextmanager

    @contextmanager
    def _first_open_cm(p: str, m: str) -> Generator[io.BytesIO, None, None]:
        f = io.BytesIO(Path(p).read_bytes())
        try:
            yield f
        finally:
            f.close()

    def _open_monkey(
        path: str,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> AbstractContextManager[io.BytesIO] | _Reader:
        call_state["n"] += 1
        if call_state["n"] == 1:
            return _first_open_cm(path, mode)
        if path.endswith("logs.jsonl") and "rb" in mode:
            return _Reader(b"two\n")
        raise AssertionError("unexpected open call")

    import model_trainer.api.routes.runs as runs_mod

    monkeypatch.setattr(runs_mod, "open", _open_monkey, raising=False)

    # Drive the iterator directly without HTTP to avoid timeouts
    gen = h._sse_iter(str(run_dir / "logs.jsonl"), tail=1, follow=True)
    out: list[bytes] = list(gen)
    # Assert: initial tail emitted, sleep branch executed once, then closed
    assert len(out) >= 1 and out[0].startswith(b"data: ")
    assert sleep_calls["n"] >= 1
