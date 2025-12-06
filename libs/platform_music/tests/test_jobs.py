from __future__ import annotations

import pytest
from platform_core.job_events import decode_job_event, is_failed
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis

from platform_music import FakeLastFm, process_wrapped_job
from platform_music.jobs import AppleMusicCredentials, SpotifyCredentials, WrappedJobPayload


def test_process_wrapped_job_success(monkeypatch: pytest.MonkeyPatch) -> None:
    # Prepare fakes
    fake_lastfm = FakeLastFm()
    for i in range(15):
        fake_lastfm.add_play(
            track_id=f"t{i}",
            title=f"S{i}",
            artist_name=f"A{i % 2}",
            played_at=f"2024-{(i % 12) + 1:02d}-01T00:00:00Z",
        )
    fake_redis = FakeRedis()

    # Monkeypatch factories
    import platform_music.jobs as jobs_mod

    def _lfm(**_: str) -> FakeLastFm:
        return fake_lastfm

    def _rf(url: str) -> FakeRedis:
        assert url == "redis://ignored"
        return fake_redis

    monkeypatch.setattr(jobs_mod, "lastfm_client", _lfm)
    monkeypatch.setattr(jobs_mod, "_redis_client", _rf)

    payload: WrappedJobPayload = {
        "type": "music_wrapped.generate.v1",
        "year": 2024,
        "service": "lastfm",
        "credentials": {"api_key": "x", "api_secret": "y", "session_key": "z"},
        "user_id": 99,
        "redis_url": "redis://ignored",
        "queue_name": "music_wrapped",
    }

    rid = process_wrapped_job(payload)
    assert rid == "wrapped:99:2024"

    # Verify result stored in Redis
    raw = fake_redis.get(rid)
    if raw is None:
        raise AssertionError("expected redis value")
    data = load_json_str(raw)
    if not isinstance(data, dict):
        raise AssertionError("expected dict json")
    doc: dict[str, JSONValue] = data
    assert doc.get("year") == 2024
    assert doc.get("service") == "lastfm"

    # Verify events were published
    types = [decode_job_event(p.payload)["type"] for p in fake_redis.published]
    assert any(t.endswith("started.v1") for t in types)
    assert any(t.endswith("completed.v1") for t in types)


def test__redis_client_wrapper(monkeypatch: pytest.MonkeyPatch) -> None:
    # Ensure wrapper calls underlying factory and returns it
    from platform_workers.testing import FakeRedis

    import platform_music.jobs as jobs_mod

    fake = FakeRedis()

    def _rf(url: str) -> FakeRedis:
        return fake

    monkeypatch.setattr(jobs_mod, "redis_for_kv", _rf)
    out = jobs_mod._redis_client("redis://ignored")
    assert out is fake


def test_process_wrapped_job_unsupported_service(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_redis = FakeRedis()

    import platform_music.jobs as jobs_mod

    def _rf(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(jobs_mod, "_redis_client", _rf)

    payload: WrappedJobPayload = {
        "type": "music_wrapped.generate.v1",
        "year": 2024,
        "service": "spotify",
        "credentials": {"api_key": "x", "api_secret": "y", "session_key": "z"},
        "user_id": 5,
        "redis_url": "redis://ignored",
        "queue_name": "music_wrapped",
    }

    from platform_core.errors import AppError

    with pytest.raises(AppError):
        process_wrapped_job(payload)

    # Verify failed event was published
    events = [decode_job_event(p.payload) for p in fake_redis.published]
    assert any(is_failed(e) for e in events)


def test_process_wrapped_job_invalid_lastfm_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_redis = FakeRedis()

    import platform_music.jobs as jobs_mod

    def _rf(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(jobs_mod, "_redis_client", _rf)

    bad_creds: SpotifyCredentials = {
        "access_token": "tok",
        "refresh_token": "rt",
        "expires_in": 3600,
    }
    payload: WrappedJobPayload = {
        "type": "music_wrapped.generate.v1",
        "year": 2024,
        "service": "lastfm",
        # Wrong credential type on purpose
        "credentials": bad_creds,
        "user_id": 5,
        "redis_url": "redis://ignored",
        "queue_name": "music_wrapped",
    }

    from platform_core.errors import AppError

    with pytest.raises(AppError):
        process_wrapped_job(payload)

    events = [decode_job_event(p.payload) for p in fake_redis.published]
    assert any(is_failed(e) for e in events)


def test_process_wrapped_job_invalid_apple_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_redis = FakeRedis()

    import platform_music.jobs as jobs_mod

    def _rf(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(jobs_mod, "_redis_client", _rf)

    bad_creds_sp: SpotifyCredentials = {
        "access_token": "tok",
        "refresh_token": "rt",
        "expires_in": 3600,
    }

    payload: WrappedJobPayload = {
        "type": "music_wrapped.generate.v1",
        "year": 2024,
        "service": "apple_music",
        # Supply Spotify credentials to Apple to force validation error
        "credentials": bad_creds_sp,
        "user_id": 5,
        "redis_url": "redis://ignored",
        "queue_name": "music_wrapped",
    }

    from platform_core.errors import AppError

    with pytest.raises(AppError):
        process_wrapped_job(payload)

    events = [decode_job_event(p.payload) for p in fake_redis.published]
    assert any(is_failed(e) for e in events)


def test_process_wrapped_job_invalid_youtube_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_redis = FakeRedis()

    import platform_music.jobs as jobs_mod

    def _rf(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(jobs_mod, "_redis_client", _rf)

    bad_creds_ap: AppleMusicCredentials = {
        "developer_token": "d",
        "music_user_token": "u",
    }

    payload: WrappedJobPayload = {
        "type": "music_wrapped.generate.v1",
        "year": 2024,
        "service": "youtube_music",
        # Supply Apple credentials to YouTube to force validation error
        "credentials": bad_creds_ap,
        "user_id": 5,
        "redis_url": "redis://ignored",
        "queue_name": "music_wrapped",
    }

    from platform_core.errors import AppError

    with pytest.raises(AppError):
        process_wrapped_job(payload)

    events = [decode_job_event(p.payload) for p in fake_redis.published]
    assert any(is_failed(e) for e in events)
