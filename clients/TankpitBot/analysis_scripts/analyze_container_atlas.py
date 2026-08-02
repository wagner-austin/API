"""Second-pass analysis of the mined container observations.

Reads ``runs/analysis/container_observations.jsonl`` (from
``mine_container_atlas.py``) and answers the persistence questions the
raw counts left open:

1. Are there WITHIN-session volume increases? (Would falsify the
   2026-07-25 in-session "nothing spawns" law.)
2. What is the refill timescale? (dt distribution of cross-session
   increases — the regeneration law's clock.)
3. How does cross-session volume agreement decay with elapsed time?
   (The atlas persistence half-life: how fresh must a snapshot be to
   seed the sim truthfully.)
4. Instantaneous vs cumulative census: distinct tiles per session vs
   the 10,930 all-time union.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

OBS = Path("runs/analysis/container_observations.jsonl")

HOUR = 3_600_000
DAY = 24 * HOUR


def _bucket(dt_ms: int) -> str:
    if dt_ms < HOUR:
        return "<1h"
    if dt_ms < DAY:
        return "1h-1d"
    if dt_ms < 3 * DAY:
        return "1d-3d"
    if dt_ms < 7 * DAY:
        return "3d-7d"
    if dt_ms < 30 * DAY:
        return "7d-30d"
    return ">30d"


def main() -> int:
    by_tile: dict[tuple, list] = defaultdict(list)
    sessions: dict[str, set] = defaultdict(set)
    t_min, t_max = None, None
    for line in OBS.read_text(encoding="utf-8").splitlines():
        t, seq, room, field, x, y, v, src, stamp = json.loads(line)
        by_tile[(room, x, y)].append((t, seq, v, src, stamp))
        if v != 0:
            sessions[stamp].add((x, y))
        t_min = t if t_min is None else min(t_min, t)
        t_max = t if t_max is None else max(t_max, t)
    assert t_min is not None and t_max is not None
    print(f"archive span: {(t_max - t_min) / DAY:.1f} days, {len(sessions)} sessions with stock")

    # Layer discipline: a 0 from the VISIBLE layer (0x5A ``v`` / 0x43
    # ``c``) says "no visible container" and nothing about hidden
    # stock — only radar zeros and pickup remaining=0 are true empty
    # statements. Increases from a visible-layer 0 are the EXPOSURE
    # law (hidden container revealed later), never refills.
    within_session_increases = 0
    within_examples = []
    refill_records = []
    exposure_pairs = 0
    refill_buckets: dict[str, int] = defaultdict(int)
    refill_dvs: list[int] = []
    agreement: dict[str, list[int]] = defaultdict(list)  # bucket -> [1 same / 0 changed]
    for key, timeline in by_tile.items():
        timeline.sort()
        prev = None  # (t, v, src, stamp) of last authoritative fuel read
        for t, _seq, v, src, stamp in timeline:
            if v < 0:
                continue
            if v == 0 and src in ("v", "c"):
                continue  # visible-layer silence, not an empty statement
            if prev is not None:
                pt, pv, psrc, pstamp = prev
                if v > pv:
                    if pv == 0 and psrc in ("r", "p") and stamp != pstamp:
                        # true-empty -> stocked across sessions: a refill
                        refill_buckets[_bucket(t - pt)] += 1
                        refill_dvs.append(v - pv)
                        refill_records.append(
                            {"kind": "cross", "room": key[0], "x": key[1], "y": key[2],
                             "pv": pv, "v": v, "pt": pt, "t": t,
                             "pstamp": pstamp, "stamp": stamp}
                        )
                    elif pv == 0:
                        exposure_pairs += 1
                    elif stamp == pstamp:
                        within_session_increases += 1
                        if len(within_examples) < 12:
                            within_examples.append((key, pv, v, src, stamp))
                        refill_records.append(
                            {"kind": "within", "room": key[0], "x": key[1], "y": key[2],
                             "pv": pv, "v": v, "pt": pt, "t": t,
                             "pstamp": pstamp, "stamp": stamp}
                        )
                    else:
                        refill_buckets[_bucket(t - pt)] += 1
                        refill_dvs.append(v - pv)
                        refill_records.append(
                            {"kind": "cross", "room": key[0], "x": key[1], "y": key[2],
                             "pv": pv, "v": v, "pt": pt, "t": t,
                             "pstamp": pstamp, "stamp": stamp}
                        )
                if stamp != pstamp and pv > 0:
                    agreement[_bucket(t - pt)].append(1 if v == pv else 0)
            prev = (t, v, src, stamp)

    print(f"\nWITHIN-session volume increases: {within_session_increases}")
    for key, pv, v, src, stamp in within_examples:
        print(f"    tile {key}: {pv} -> {v} (src {src}) in {stamp}")

    print("\nCross-session REFILL dt distribution:")
    for bucket in ("<1h", "1h-1d", "1d-3d", "3d-7d", "7d-30d", ">30d"):
        print(f"    {bucket:>7}: {refill_buckets.get(bucket, 0)}")
    if refill_dvs:
        refill_dvs.sort()
        mid = refill_dvs[len(refill_dvs) // 2]
        print(f"    dv median {mid}, min {refill_dvs[0]}, max {refill_dvs[-1]}")

    print("\nCross-session volume AGREEMENT rate by elapsed time (fuel>0 pairs):")
    for bucket in ("<1h", "1h-1d", "1d-3d", "3d-7d", "7d-30d", ">30d"):
        pairs = agreement.get(bucket, [])
        if pairs:
            rate = 100.0 * sum(pairs) / len(pairs)
            print(f"    {bucket:>7}: {rate:5.1f}% same volume  (n={len(pairs)})")

    per_session = sorted(len(tiles) for tiles in sessions.values())
    mid = per_session[len(per_session) // 2]
    print(
        f"\nstocked tiles observed per session: median {mid}, max {per_session[-1]} "
        f"(cumulative union {sum(1 for tl in by_tile.values() if any(r[2] != 0 for r in tl))})"
    )

    # Freshness snapshot: how big is the atlas if we only trust the
    # most recent N days of observations?
    Path("runs/analysis/container_refills.json").write_text(
        json.dumps(refill_records, indent=1), encoding="utf-8"
    )
    print(f"wrote runs/analysis/container_refills.json ({len(refill_records)} events)")

    for days in (7, 30):
        cutoff = t_max - days * DAY
        fresh = {
            key
            for key, timeline in by_tile.items()
            if any(t >= cutoff and v != 0 for t, _, v, _, _ in timeline)
        }
        print(f"tiles with stocked reads in the last {days} days: {len(fresh)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
