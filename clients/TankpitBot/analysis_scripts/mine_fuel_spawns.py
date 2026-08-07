"""Archive sweep: fuel-container spawn/despawn dynamics from 0x4C fuel-dot atlas diffs.

Migrated 2026-08-06 onto ``tankpit_bot.analysis.scan`` (the typed
capture-scan owner) - the private load/XOR/frame-walk pipeline is
deleted; results reproduce exactly.
"""

from collections import Counter
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol import decode_message

agg = Counter()
gap_appear_rate = Counter()  # appearances bucketed by inter-snapshot gap
quadrant_appear = Counter()
dot_counts = []
appear_events = 0
disappear_events = 0
observed_map_minutes = 0.0
back_to_back_diffs = Counter()

for path in sorted(Path("runs").glob("*/*.capture_session.json")):
    try:
        result = scan_session(path)
    except Exception:
        continue
    if "reason" in result:
        continue
    prev_dots = None
    prev_ts = None
    for frame in sorted(result["frames"], key=lambda f: f["timestamp_ms"]):
        if frame["direction"] != "received":
            continue
        ts = frame["timestamp_ms"]
        try:
            dm = decode_message(frame["msg_type"], frame["body"])
        except Exception:
            continue
        if dm["msg_type"] != 0x4C:
            continue
        dots = frozenset(tuple(d) for d in dm["fuel_dots"])
        agg["snapshots"] += 1
        dot_counts.append(len(dots))
        if prev_dots is not None:
            gap_s = (ts - prev_ts) / 1000
            appeared = dots - prev_dots
            disappeared = prev_dots - dots
            if gap_s <= 5:
                back_to_back_diffs[min(len(appeared) + len(disappeared), 10)] += 1
            else:
                appear_events += len(appeared)
                disappear_events += len(disappeared)
                observed_map_minutes += gap_s / 60
                bucket = min(int(gap_s // 30), 10)
                gap_appear_rate[bucket * 30] += len(appeared)
                for x, y in appeared:
                    quadrant_appear[(x // 64, y // 64)] += 1
        prev_dots = dots
        prev_ts = ts

dot_counts.sort()
n = len(dot_counts)
print(
    f"snapshots: {agg['snapshots']}  dot-count median {dot_counts[n // 2] if n else 0} "
    f"p10 {dot_counts[n // 10] if n else 0} p90 {dot_counts[9 * n // 10] if n else 0}"
)
print(f"back-to-back (<=5 s) diffs: {dict(sorted(back_to_back_diffs.items()))}")
print(
    f"appearances: {appear_events}  disappearances: {disappear_events} "
    f"over {observed_map_minutes:.0f} map-open-gap minutes"
)
if observed_map_minutes:
    print(
        f"MAP-WIDE spawn rate: {appear_events / observed_map_minutes:.2f} dots/min "
        f"(65,536 tiles -> {appear_events / observed_map_minutes / 65536 * 1e6:.1f} spawns per million tile-min)"
    )
print("appearances by inter-open gap bucket (s):", dict(sorted(gap_appear_rate.items())))
qa = Counter()
for (qx, qy), c in quadrant_appear.items():
    qa[f"{qx},{qy}"] = c
print("appearance quadrants (4x4 of 64x64):", dict(sorted(qa.items())))
