from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Literal, Protocol

from _pytest.monkeypatch import MonkeyPatch
from fastapi.testclient import TestClient

from model_trainer.api.main import create_app
from model_trainer.core.config.settings import Settings


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


def test_runs_logs_stream_handles_oserror_and_ends(
    tmp_path: Path, monkeypatch: MonkeyPatch, settings_factory: _SettingsFactory
) -> None:
    # Arrange artifacts and a run log with two lines
    artifacts = tmp_path / "artifacts"
    run_id = "run-err"
    log_dir = artifacts / "models" / run_id
    log_dir.mkdir(parents=True)
    log_path = log_dir / "logs.jsonl"
    log_path.write_text("one\ntwo\n", encoding="utf-8")

    app = create_app(settings_factory(artifacts_root=str(artifacts)))
    client = TestClient(app)

    # Patch open so that the second open (follow phase) raises OSError
    from pathlib import Path as PathAlias

    calls: dict[str, int] = {"count": 0}

    def fake_open(
        file: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ) -> io.BytesIO:
        path_s = str(file)
        if path_s.endswith("logs.jsonl") and "rb" in mode:
            calls["count"] += 1
            if calls["count"] >= 2:
                raise OSError("boom")
            data = PathAlias(path_s).read_bytes()
            return io.BytesIO(data)
        raise AssertionError("unexpected open mode or path in test stub")

    monkeypatch.setattr("model_trainer.api.routes.runs.open", fake_open, raising=False)

    # Act: stream (default follow=True). The generator will emit tail lines
    # then hit OSError and terminate. Collect what we can and ensure it ends.
    with client.stream("GET", f"/runs/{run_id}/logs/stream", params={"tail": 1}) as resp:
        assert resp.status_code == 200
        chunks = list(resp.iter_bytes())

    # Assert: at least one SSE data line came through and stream ended quickly
    body = b"".join(chunks)
    assert b"data: " in body
    # And ensure our error path was triggered (second open attempted)
    assert calls["count"] >= 2
