"""Archive check: radar low-fuel no-debit floor + initial extras stock."""
import json
from collections import Counter
from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict

from tankpit_bot.types import decode_capture_session
from tankpit_bot.validate.wire_timeline import extract_wire_timeline
from tankpit_bot.protocol.commands import CMD_RADAR

first_inventory = Counter()
floor_deltas = Counter()      # fuel-level bucket -> delta counter for isolated radar windows
low_fuel_examples = []

for path in sorted(Path("runs").glob("*/*.capture_session.json")):
    try:
        session = decode_capture_session(
            narrow_json_to_dict(load_json_str(path.read_text(encoding="utf-8")))
        )
    except Exception:
        continue
    if session["magic"] is None:
        continue
    tl = extract_wire_timeline(session)
    if tl["inventory_snapshots"]:
        first_inventory[tl["inventory_snapshots"][0]["counts"][4]] += 1
    readings = tl["fuel_readings"]
    radar_times = sorted(
        a["timestamp_ms"] for a in tl["sent_actions"] if a["command"] == CMD_RADAR
    )
    other_times = sorted(
        a["timestamp_ms"] for a in tl["sent_actions"] if a["command"] != CMD_RADAR
    )
    contamination = sorted(
        [s["timestamp_ms"] for s in tl["own_shots"]]
        + [s["timestamp_ms"] for s in tl["enemy_shots"]]
        + tl["pickup_timestamps"]
        + tl["detonation_timestamps"]
        + [r["timestamp_ms"] for r in readings if r["from_event"]]
    )
    from bisect import bisect_right
    for i in range(1, len(readings)):
        a, b = readings[i - 1], readings[i]
        ta, tb = a["timestamp_ms"], b["timestamp_ms"]
        radars = bisect_right(radar_times, tb) - bisect_right(radar_times, ta)
        others = bisect_right(other_times, tb) - bisect_right(other_times, ta)
        dirty = bisect_right(contamination, tb) - bisect_right(contamination, ta - 3000)
        if radars != 1 or others or dirty:
            continue
        delta = b["fuel"] - a["fuel"]
        bucket = "0-49" if a["fuel"] < 50 else "50-99" if a["fuel"] < 100 else "100-199" if a["fuel"] < 200 else "200+"
        floor_deltas[(bucket, delta)] += 1
        if a["fuel"] < 100 and len(low_fuel_examples) < 10:
            low_fuel_examples.append((path.name[:28], a["fuel"], delta))

print("first 0x49 radar count (slot 4) per session:", dict(sorted(first_inventory.items())))
print("\nisolated radar windows: (fuel-bucket, delta) -> count")
for (bucket, delta), c in sorted(floor_deltas.items()):
    print(f"  {bucket:>8} delta {delta:>4}: {c}")
print("\nlow-fuel examples:", low_fuel_examples)
