from __future__ import annotations

from typing import Never

import pytest
from platform_core.job_events import decode_job_event, is_failed
from platform_workers.testing import FakeRedis

from platform_music.jobs import WrappedJobPayload, process_wrapped_job


def test_process_wrapped_job_system_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_redis = FakeRedis()

    import platform_music.jobs as jobs_mod

    def _rf(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(jobs_mod, "_redis_client", _rf)

    # Force a system error during generation
    import platform_music.wrapped as wrapped_mod
    from platform_music.services.protocol import MusicServiceProto

    class _BoomGen:
        def __init__(self, music_client: MusicServiceProto) -> None:
            _ = music_client

        def generate_wrapped(self, *, year: int) -> Never:
            raise RuntimeError("boom")

    monkeypatch.setattr(wrapped_mod, "WrappedGenerator", _BoomGen)

    payload: WrappedJobPayload = {
        "type": "music_wrapped.generate.v1",
        "year": 2024,
        "service": "lastfm",
        "credentials": {"api_key": "x", "api_secret": "y", "session_key": "z"},
        "user_id": 7,
        "redis_url": "redis://ignored",
        "queue_name": "music_wrapped",
    }

    with pytest.raises(RuntimeError):
        process_wrapped_job(payload)

    events = [decode_job_event(p.payload) for p in fake_redis.published]
    assert any(is_failed(e) for e in events)
    fake_redis.assert_only_called({"publish"})
