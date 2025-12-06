from __future__ import annotations

from pathlib import Path
from typing import Literal, Protocol

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


def test_runs_logs_stream_initial_tail(tmp_path: Path, settings_factory: _SettingsFactory) -> None:
    artifacts = tmp_path / "artifacts"
    run_id = "run-stream"
    log_dir = artifacts / "models" / run_id
    log_dir.mkdir(parents=True)
    lines = [b'{"msg":"one"}\n', b'{"msg":"two"}\n', b'{"msg":"three"}\n']
    (log_dir / "logs.jsonl").write_bytes(b"".join(lines))

    app = create_app(settings_factory(artifacts_root=str(artifacts)))
    client = TestClient(app)

    # Act: stream with tail=2 and collect only the first two SSE lines
    collected: list[str] = []
    with client.stream(
        "GET",
        f"/runs/{run_id}/logs/stream",
        params={"tail": 2, "follow": False},
    ) as r:
        assert r.status_code == 200
        for raw in r.iter_lines():
            if not raw:
                continue
            if isinstance(raw, bytes):
                b: bytes = raw
                text = b.decode("utf-8", errors="ignore")
            else:
                text = raw
            if text.startswith("data: "):
                collected.append(text[len("data: ") :])
                if len(collected) >= 2:
                    break

    # Assert: last two lines are emitted first in order
    assert len(collected) == 2
    assert '{"msg":"two"}' in collected[0]
    assert '{"msg":"three"}' in collected[1]
