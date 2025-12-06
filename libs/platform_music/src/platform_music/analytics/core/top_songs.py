from __future__ import annotations

from collections import Counter

from platform_music.models import PlayRecord, TopSong


def compute_top_songs(plays: list[PlayRecord], *, limit: int = 5) -> list[TopSong]:
    counts: Counter[tuple[str, str]] = Counter(
        (p["track"]["title"], p["track"]["artist_name"]) for p in plays
    )
    ranked: list[TopSong] = [
        {"title": title, "artist_name": artist, "play_count": count}
        for (title, artist), count in counts.most_common(limit)
    ]
    return ranked


__all__ = ["compute_top_songs"]
