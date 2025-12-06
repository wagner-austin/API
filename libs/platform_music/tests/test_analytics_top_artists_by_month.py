from __future__ import annotations

from platform_music.analytics.core.top_artists_by_month import compute_top_artists_by_month
from platform_music.testing import FakeLastFm


def test_compute_top_artists_by_month() -> None:
    fake = FakeLastFm()
    # Jan: A(2), B(1); Feb: A(1)
    fake.add_play(track_id="a1", title="X", artist_name="A", played_at="2024-01-01T00:00:00Z")
    fake.add_play(track_id="a2", title="X", artist_name="A", played_at="2024-01-02T00:00:00Z")
    fake.add_play(track_id="b1", title="Y", artist_name="B", played_at="2024-01-03T00:00:00Z")
    fake.add_play(track_id="a3", title="X", artist_name="A", played_at="2024-02-01T00:00:00Z")
    # Non-Z ISO timezone form
    fake.add_play(track_id="a4", title="X", artist_name="A", played_at="2024-02-02T00:00:00+00:00")

    plays = fake.get_listening_history(
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-12-31T23:59:59Z",
        limit=None,
    )
    top = compute_top_artists_by_month(plays, limit=2)
    assert [e["month"] for e in top] == [1, 2]
    jan = top[0]
    assert [a["artist_name"] for a in jan["top_artists"]] == ["A", "B"]
    assert [a["play_count"] for a in jan["top_artists"]] == [2, 1]
