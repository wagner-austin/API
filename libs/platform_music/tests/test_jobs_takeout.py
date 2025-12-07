from __future__ import annotations

from platform_core.json_utils import JSONValue, dump_json_str, load_json_str
from platform_workers.testing import FakeRedis
from pytest import MonkeyPatch

from platform_music.jobs import ImportYouTubeTakeoutJobPayload, process_import_youtube_takeout


def test_process_import_youtube_takeout_success(monkeypatch: MonkeyPatch) -> None:
    # Prepare plays for 2024 (>= DEFAULT_MIN_PLAYS = 10)
    plays: list[dict[str, JSONValue]] = []
    for i in range(15):
        plays.append(
            {
                "track": {
                    "id": f"v{i}",
                    "title": f"S{i}",
                    "artist_name": f"A{i % 2}",
                    "duration_ms": 0,
                    "service": "youtube_music",
                },
                "played_at": f"2024-{(i % 12) + 1:02d}-01T00:00:00Z",
                "service": "youtube_music",
            }
        )

    fr = FakeRedis()
    token_id = "tok"
    fr.set(f"ytmusic:takeout:{token_id}", dump_json_str(plays))

    import platform_music.jobs as jobs_mod

    def _rf(url: str) -> FakeRedis:
        assert url == "redis://ignored"
        return fr

    monkeypatch.setattr(jobs_mod, "_redis_client", _rf)

    payload: ImportYouTubeTakeoutJobPayload = {
        "type": "music_wrapped.import_youtube_takeout.v1",
        "year": 2024,
        "token_id": token_id,
        "user_id": 7,
        "redis_url": "redis://ignored",
        "queue_name": "music_wrapped",
    }

    rid = process_import_youtube_takeout(payload)
    assert rid == "wrapped:7:2024"
    raw = fr.get(rid)
    if raw is None:
        raise AssertionError("expected result in redis")
    doc = load_json_str(raw)
    if not isinstance(doc, dict):
        raise AssertionError("result must be object")
    assert doc.get("service") == "youtube_music"
    assert doc.get("year") == 2024
    fr.assert_only_called({"set", "get", "publish", "delete"})


def test_process_import_youtube_takeout_missing(monkeypatch: MonkeyPatch) -> None:
    fr = FakeRedis()

    import platform_music.jobs as jobs_mod

    def _rf(url: str) -> FakeRedis:
        return fr

    monkeypatch.setattr(jobs_mod, "_redis_client", _rf)

    payload: ImportYouTubeTakeoutJobPayload = {
        "type": "music_wrapped.import_youtube_takeout.v1",
        "year": 2024,
        "token_id": "nope",
        "user_id": 1,
        "redis_url": "redis://ignored",
        "queue_name": "music_wrapped",
    }

    import pytest
    from platform_core.errors import AppError

    with pytest.raises(AppError) as excinfo:
        process_import_youtube_takeout(payload)
    assert excinfo.value.http_status == 404
    fr.assert_only_called({"get", "publish"})
