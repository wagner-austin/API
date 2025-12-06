from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime

from platform_music.models import PlayRecord, TopArtist, TopArtistsByMonthEntry


def _parse_iso(ts: str) -> datetime:
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def compute_top_artists_by_month(
    plays: list[PlayRecord], *, limit: int = 3
) -> list[TopArtistsByMonthEntry]:
    buckets: dict[int, Counter[str]] = defaultdict(Counter)
    for p in plays:
        month = _parse_iso(p["played_at"]).month
        buckets[month][p["track"]["artist_name"]] += 1
    out: list[TopArtistsByMonthEntry] = []
    for month in sorted(buckets.keys()):
        ranked: list[TopArtist] = [
            {"artist_name": name, "play_count": count}
            for name, count in buckets[month].most_common(limit)
        ]
        out.append({"month": month, "top_artists": ranked})
    return out


__all__ = ["compute_top_artists_by_month"]
