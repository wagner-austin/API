from __future__ import annotations

from platform_music.analytics.core.top_songs import compute_top_songs
from platform_music.testing import FakeLastFm


def test_compute_top_songs_counts_and_order() -> None:
    fake = FakeLastFm()
    # Add plays: Song A (3), Song B (2), Song C (1)
    fake.add_play(track_id="a1", title="A", artist_name="X", played_at="2024-01-01T00:00:00Z")
    fake.add_play(track_id="a2", title="A", artist_name="X", played_at="2024-01-02T00:00:00Z")
    fake.add_play(track_id="a3", title="A", artist_name="X", played_at="2024-01-03T00:00:00Z")
    fake.add_play(track_id="b1", title="B", artist_name="Y", played_at="2024-01-04T00:00:00Z")
    fake.add_play(track_id="b2", title="B", artist_name="Y", played_at="2024-01-05T00:00:00Z")
    fake.add_play(track_id="c1", title="C", artist_name="Z", played_at="2024-01-06T00:00:00Z")

    plays = fake.get_listening_history(
        start_date="2024-01-01T00:00:00Z",
        end_date="2024-12-31T23:59:59Z",
        limit=None,
    )
    top = compute_top_songs(plays, limit=3)
    assert [t["title"] for t in top] == ["A", "B", "C"]
    assert [t["play_count"] for t in top] == [3, 2, 1]
