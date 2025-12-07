from __future__ import annotations

from pathlib import Path

from _pytest.monkeypatch import MonkeyPatch
from platform_workers.redis import RedisStrProto
from platform_workers.testing import FakeRedis

from model_trainer.api.routes.runs import _RunsRoutes
from model_trainer.core.config.settings import load_settings
from model_trainer.core.services.container import ServiceContainer


def test_runs_logs_stream_follow_none_max_loops_exercises_else_branch(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    # Real container instance with patched redis_for_kv to use _FakeRedis
    fake = FakeRedis()

    def _fake_redis_for_kv(url: str) -> RedisStrProto:
        return fake

    monkeypatch.setattr("model_trainer.core.services.container.redis_for_kv", _fake_redis_for_kv)
    s = load_settings()
    container: ServiceContainer = ServiceContainer.from_settings(s)

    # Prepare a log file with a single line
    log_path = tmp_path / "logs.jsonl"
    log_path.write_text("one\n", encoding="utf-8")

    # Create the routes helper with real container for logging
    h = _RunsRoutes(container)

    # Provide a sleep function that appends a line to the file to wake the follower
    def _sleep_and_append(_: float) -> None:
        with open(log_path, "ab") as f:
            f.write(b"two\n")

    h._sleep_fn = _sleep_and_append
    h._follow_max_loops = None  # ensure the None branch at line 110 is evaluated

    # Consume SSE iterator: first yield is initial tail; second after sleep/appended line
    it = h._sse_iter(str(log_path), tail=1, follow=True)
    first = next(it)  # initial tail ("one")
    assert first.startswith(b"data: ")
    second = next(it)  # after sleep/appended line ("two")
    assert second.startswith(b"data: ")
    fake.assert_only_called(set())
