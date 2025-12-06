from __future__ import annotations

from platform_core.queues import MUSIC_WRAPPED_QUEUE
from pytest import MonkeyPatch


def test_worker_build_config(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://unit-test")
    import music_wrapped_api.worker_entry as we

    cfg = we._build_config()
    assert cfg["redis_url"] == "redis://unit-test"
    assert cfg["queue_name"] == MUSIC_WRAPPED_QUEUE
    assert cfg["events_channel"].endswith(":events")


def test_worker_main_invokes_runner(monkeypatch: MonkeyPatch) -> None:
    monkeypatch.setenv("REDIS_URL", "redis://unit-test")
    called: dict[str, str] = {}

    import music_wrapped_api.worker_entry as we

    def _run(cfg: dict[str, str]) -> None:
        called["queue"] = cfg["queue_name"]
        called["events"] = cfg["events_channel"]

    monkeypatch.setattr(we, "run_rq_worker", _run)
    we.main()

    assert called["queue"] == MUSIC_WRAPPED_QUEUE
    assert called["events"].startswith("music_wrapped:")
