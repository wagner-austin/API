from __future__ import annotations

from platform_core.job_events import decode_job_event
from platform_core.json_utils import load_json_str
from platform_workers.testing import FakeRedis

from platform_music.jobs import WrappedJobPayload, process_wrapped_job
from platform_music.models import ServiceName
from platform_music.testing import (
    FakeAppleMusic,
    FakeSpotify,
    FakeYouTubeMusic,
    hooks,
    make_fake_apple_client,
    make_fake_redis_client,
    make_fake_spotify_client,
    make_fake_youtube_client,
)


def _populate_fake(
    fake: FakeSpotify | FakeAppleMusic | FakeYouTubeMusic, service: ServiceName
) -> None:
    """Add 12 plays to a fake music service."""
    for i in range(12):
        fake.add_play(
            track_id=f"{service}:t{i}",
            title=f"S{i}",
            artist_name=f"A{i % 3}",
            played_at=f"2024-{(i % 12) + 1:02d}-01T00:00:00Z",
            duration_ms=1000,
        )


def test_process_wrapped_job_spotify_success() -> None:
    fake_redis = FakeRedis()
    fake_spotify = FakeSpotify()
    _populate_fake(fake_spotify, "spotify")

    hooks.redis_client = make_fake_redis_client(fake_redis)
    hooks.spotify_client = make_fake_spotify_client(fake_spotify)

    p_sp: WrappedJobPayload = {
        "type": "music_wrapped.generate.v1",
        "year": 2024,
        "service": "spotify",
        "credentials": {"access_token": "at", "refresh_token": "rt", "expires_in": 3600},
        "user_id": 1,
        "redis_url": "redis://ignored",
        "queue_name": "music_wrapped",
    }
    rid_sp = process_wrapped_job(p_sp)
    raw_sp = fake_redis.get(rid_sp)
    if not isinstance(raw_sp, str):
        raise AssertionError("expected string json")
    data_sp = load_json_str(raw_sp)
    if not isinstance(data_sp, dict):
        raise AssertionError("expected dict json")
    assert data_sp.get("service") == "spotify"
    fake_redis.assert_only_called({"set", "get", "publish"})


def test_process_wrapped_job_apple_success() -> None:
    fake_redis = FakeRedis()
    fake_apple = FakeAppleMusic()
    _populate_fake(fake_apple, "apple_music")

    hooks.redis_client = make_fake_redis_client(fake_redis)
    hooks.apple_client = make_fake_apple_client(fake_apple)

    p_ap: WrappedJobPayload = {
        "type": "music_wrapped.generate.v1",
        "year": 2024,
        "service": "apple_music",
        "credentials": {"developer_token": "d", "music_user_token": "u"},
        "user_id": 2,
        "redis_url": "redis://ignored",
        "queue_name": "music_wrapped",
    }
    rid_ap = process_wrapped_job(p_ap)
    raw_ap = fake_redis.get(rid_ap)
    if not isinstance(raw_ap, str):
        raise AssertionError("expected string json")
    data_ap = load_json_str(raw_ap)
    if not isinstance(data_ap, dict):
        raise AssertionError("expected dict json")
    assert data_ap.get("service") == "apple_music"
    fake_redis.assert_only_called({"set", "get", "publish"})


def test_process_wrapped_job_youtube_success() -> None:
    fake_redis = FakeRedis()
    fake_youtube = FakeYouTubeMusic()
    _populate_fake(fake_youtube, "youtube_music")

    hooks.redis_client = make_fake_redis_client(fake_redis)
    hooks.youtube_client = make_fake_youtube_client(fake_youtube)

    p_yt: WrappedJobPayload = {
        "type": "music_wrapped.generate.v1",
        "year": 2024,
        "service": "youtube_music",
        "credentials": {"sapisid": "sid", "cookies": "a=b"},
        "user_id": 3,
        "redis_url": "redis://ignored",
        "queue_name": "music_wrapped",
    }
    rid_yt = process_wrapped_job(p_yt)
    raw_yt = fake_redis.get(rid_yt)
    if not isinstance(raw_yt, str):
        raise AssertionError("expected string json")
    data_yt = load_json_str(raw_yt)
    if not isinstance(data_yt, dict):
        raise AssertionError("expected dict json")
    assert data_yt.get("service") == "youtube_music"

    # Verify events include completed for one of them at least
    types = [decode_job_event(p.payload)["type"] for p in fake_redis.published]
    assert any(t.endswith("completed.v1") for t in types)
    fake_redis.assert_only_called({"set", "get", "publish"})
