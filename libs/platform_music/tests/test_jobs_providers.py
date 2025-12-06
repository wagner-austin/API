from __future__ import annotations

from platform_core.job_events import decode_job_event
from platform_core.json_utils import load_json_str
from platform_workers.testing import FakeRedis
from pytest import MonkeyPatch

from platform_music.jobs import WrappedJobPayload, process_wrapped_job
from platform_music.models import PlayRecord, ServiceName


def _plays(service: ServiceName) -> list[PlayRecord]:
    out: list[PlayRecord] = []
    for i in range(12):
        out.append(
            {
                "track": {
                    "id": f"{service}:t{i}",
                    "title": f"S{i}",
                    "artist_name": f"A{i % 3}",
                    "duration_ms": 1000,
                    "service": service,
                },
                "played_at": f"2024-{(i % 12) + 1:02d}-01T00:00:00Z",
                "service": service,
            }
        )
    return out


def test_process_wrapped_job_spotify_success(monkeypatch: MonkeyPatch) -> None:
    fake_redis = FakeRedis()

    import platform_music.jobs as jobs_mod

    def _rf(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(jobs_mod, "_redis_client", _rf)

    # Spotify
    import platform_music.services.spotify as sp

    class _Sp(sp.SpotifyProto):
        def get_listening_history(
            self, *, start_date: str, end_date: str, limit: int | None = None
        ) -> list[PlayRecord]:
            return _plays("spotify")

    def _mk_spotify_client(**kwargs: str | int) -> sp.SpotifyProto:
        return _Sp()

    monkeypatch.setattr(jobs_mod, "spotify_client", _mk_spotify_client)
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


def test_process_wrapped_job_apple_success(monkeypatch: MonkeyPatch) -> None:
    fake_redis = FakeRedis()

    import platform_music.jobs as jobs_mod

    def _rf(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(jobs_mod, "_redis_client", _rf)

    # Apple
    import platform_music.services.apple as ap

    class _Ap(ap.AppleMusicProto):
        def get_listening_history(
            self, *, start_date: str, end_date: str, limit: int | None = None
        ) -> list[PlayRecord]:
            return _plays("apple_music")

    def _mk_apple_client(**kwargs: str | int) -> ap.AppleMusicProto:
        return _Ap()

    monkeypatch.setattr(jobs_mod, "apple_client", _mk_apple_client)
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


def test_process_wrapped_job_youtube_success(monkeypatch: MonkeyPatch) -> None:
    fake_redis = FakeRedis()

    import platform_music.jobs as jobs_mod

    def _rf(url: str) -> FakeRedis:
        return fake_redis

    monkeypatch.setattr(jobs_mod, "_redis_client", _rf)

    # YouTube
    import platform_music.services.youtube as yt

    class _Yt(yt.YouTubeMusicProto):
        def get_listening_history(
            self, *, start_date: str, end_date: str, limit: int | None = None
        ) -> list[PlayRecord]:
            return _plays("youtube_music")

    def _mk_youtube_client(**kwargs: str | int) -> yt.YouTubeMusicProto:
        return _Yt()

    monkeypatch.setattr(jobs_mod, "youtube_client", _mk_youtube_client)
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
