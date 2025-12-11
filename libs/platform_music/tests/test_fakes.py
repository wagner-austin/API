from __future__ import annotations

from pathlib import Path

from platform_music.testing import (
    FakeAppleMusic,
    FakeSpotify,
    FakeYouTubeMusic,
    make_fake_load_orchestrator,
    make_plays,
)


def test_fake_spotify_limit_and_range() -> None:
    fake = FakeSpotify()
    fake.add_play(track_id="a", title="S1", artist_name="A1", played_at="2024-01-01T00:00:00Z")
    fake.add_play(track_id="b", title="S2", artist_name="A2", played_at="2024-06-01T00:00:00Z")
    fake.add_play(track_id="c", title="S3", artist_name="A3", played_at="2024-12-31T23:59:59Z")

    plays_all = fake.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=None
    )
    assert len(plays_all) == 3

    plays_limited = fake.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=2
    )
    assert len(plays_limited) == 2


def test_fake_apple_music_limit_and_range() -> None:
    fake = FakeAppleMusic()
    fake.add_play(track_id="a", title="S1", artist_name="A1", played_at="2024-01-01T00:00:00Z")
    fake.add_play(track_id="b", title="S2", artist_name="A2", played_at="2024-06-01T00:00:00Z")
    fake.add_play(track_id="c", title="S3", artist_name="A3", played_at="2024-12-31T23:59:59Z")

    plays_all = fake.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=None
    )
    assert len(plays_all) == 3

    plays_limited = fake.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=2
    )
    assert len(plays_limited) == 2


def test_fake_youtube_music_limit_and_range() -> None:
    fake = FakeYouTubeMusic()
    fake.add_play(track_id="a", title="S1", artist_name="A1", played_at="2024-01-01T00:00:00Z")
    fake.add_play(track_id="b", title="S2", artist_name="A2", played_at="2024-06-01T00:00:00Z")
    fake.add_play(track_id="c", title="S3", artist_name="A3", played_at="2024-12-31T23:59:59Z")

    plays_all = fake.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=None
    )
    assert len(plays_all) == 3

    plays_limited = fake.get_listening_history(
        start_date="2024-01-01T00:00:00Z", end_date="2024-12-31T23:59:59Z", limit=2
    )
    assert len(plays_limited) == 2


def test_make_plays_utility() -> None:
    plays = make_plays("spotify", count=5)
    assert len(plays) == 5
    for p in plays:
        assert p["service"] == "spotify"
        assert p["track"]["service"] == "spotify"


def test_make_fake_load_orchestrator_utility() -> None:
    loader = make_fake_load_orchestrator(exit_code=42)
    runner = loader(Path("/fake"))
    result = runner(Path("/mono"), Path("/proj"))
    assert result == 42
