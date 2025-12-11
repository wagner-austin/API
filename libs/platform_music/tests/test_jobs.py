from __future__ import annotations

import pytest
from platform_core.job_events import decode_job_event, is_failed
from platform_core.json_utils import JSONValue, load_json_str
from platform_workers.testing import FakeRedis

from platform_music import FakeLastFm, process_wrapped_job
from platform_music.jobs import AppleMusicCredentials, SpotifyCredentials, WrappedJobPayload
from platform_music.testing import hooks, make_fake_lastfm_client, make_fake_redis_client


def test_process_wrapped_job_success() -> None:
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

    # Set hooks
    hooks.lastfm_client = make_fake_lastfm_client(fake_lastfm)
    hooks.redis_client = make_fake_redis_client(fake_redis)

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

    fake_redis.assert_only_called({"set", "get", "publish"})


def test__redis_client_wrapper() -> None:
    # Ensure wrapper calls underlying factory and returns it via hook
    fake = FakeRedis()
    hooks.redis_client = make_fake_redis_client(fake)

    from platform_music.jobs import _redis_client

    out = _redis_client("redis://ignored")
    assert out is fake
    fake.assert_only_called(set())


def test_process_wrapped_job_unsupported_service() -> None:
    fake_redis = FakeRedis()
    hooks.redis_client = make_fake_redis_client(fake_redis)

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
    fake_redis.assert_only_called({"publish"})


def test_process_wrapped_job_invalid_lastfm_credentials() -> None:
    fake_redis = FakeRedis()
    hooks.redis_client = make_fake_redis_client(fake_redis)

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
    fake_redis.assert_only_called({"publish"})


def test_process_wrapped_job_invalid_apple_credentials() -> None:
    fake_redis = FakeRedis()
    hooks.redis_client = make_fake_redis_client(fake_redis)

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
    fake_redis.assert_only_called({"publish"})


def test_process_wrapped_job_invalid_youtube_credentials() -> None:
    fake_redis = FakeRedis()
    hooks.redis_client = make_fake_redis_client(fake_redis)

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
    fake_redis.assert_only_called({"publish"})
