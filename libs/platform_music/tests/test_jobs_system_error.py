from __future__ import annotations

from typing import Never

import pytest
from platform_core.job_events import decode_job_event, is_failed
from platform_workers.testing import FakeRedis

from platform_music.jobs import WrappedJobPayload, process_wrapped_job
from platform_music.services.lastfm import LastFmProto
from platform_music.testing import hooks, make_fake_redis_client


class _BoomLastFm(LastFmProto):
    """LastFm client that throws on get_listening_history."""

    def get_listening_history(
        self, *, start_date: str, end_date: str, limit: int | None = None
    ) -> Never:
        raise RuntimeError("boom")


def test_process_wrapped_job_system_error() -> None:
    fake_redis = FakeRedis()
    hooks.redis_client = make_fake_redis_client(fake_redis)

    # Return a client that throws RuntimeError
    def _boom_lastfm(api_key: str, api_secret: str, session_key: str) -> _BoomLastFm:
        return _BoomLastFm()

    hooks.lastfm_client = _boom_lastfm

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
