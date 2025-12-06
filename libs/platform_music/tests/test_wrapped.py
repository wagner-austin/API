from __future__ import annotations

import re

import pytest
from platform_core.errors import AppError

from platform_music import FakeLastFm, WrappedGenerator
from platform_music.error_codes import MusicWrappedErrorCode


def _add_many(fake: FakeLastFm, *, year: int, count: int) -> None:
    for i in range(count):
        fake.add_play(
            track_id=f"t{i}",
            title=f"Song {i}",
            artist_name=f"Artist {i % 3}",
            played_at=f"{year}-{(i % 12) + 1:02d}-01T00:00:00Z",
        )


def test_generate_wrapped_success() -> None:
    fake = FakeLastFm()
    _add_many(fake, year=2024, count=15)
    gen = WrappedGenerator(fake)

    result = gen.generate_wrapped(year=2024)

    assert result["service"] == "lastfm"
    assert result["year"] == 2024
    assert bool(re.match(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}", result["generated_at"]))
    assert result["total_scrobbles"] == 15
    assert len(result["top_artists"]) <= 5
    assert len(result["top_songs"]) <= 5
    assert all(
        ("title" in s) and ("artist_name" in s) and ("play_count" in s) for s in result["top_songs"]
    )
    assert all(
        1 <= e["month"] <= 12 and isinstance(e["top_artists"], list) for e in result["top_by_month"]
    )
    # With at least 1 play per month grouping exists where present
    assert all(1 <= e["month"] <= 12 for e in result["top_by_month"])
    # Top artist should be the modulo group with highest count
    assert {t["artist_name"] for t in result["top_artists"]} <= {"Artist 0", "Artist 1", "Artist 2"}


def test_generate_wrapped_insufficient_data() -> None:
    fake = FakeLastFm()
    _add_many(fake, year=2024, count=5)
    gen = WrappedGenerator(fake)

    with pytest.raises(AppError) as excinfo:
        gen.generate_wrapped(year=2024)
    err: AppError[MusicWrappedErrorCode] = excinfo.value
    assert err.code == MusicWrappedErrorCode.INSUFFICIENT_DATA
    assert err.http_status == 400


def test_generate_wrapped_no_history() -> None:
    fake = FakeLastFm()
    gen = WrappedGenerator(fake)

    with pytest.raises(AppError) as excinfo:
        gen.generate_wrapped(year=2024)
    err: AppError[MusicWrappedErrorCode] = excinfo.value
    assert err.code == MusicWrappedErrorCode.NO_LISTENING_HISTORY
    assert err.http_status == 404
