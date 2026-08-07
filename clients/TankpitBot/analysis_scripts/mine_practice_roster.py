"""Archive sweep: extract real practice-room layouts for sim seeding.

For each archived capture, take the client id (first 0x21 TankInfo)
and the FIRST 0x4C map snapshot: every bot slot (tank_id 500-599)
with its team, rank, and position, plus the client's own map entry
(the join spawn). Sessions with a full-size roster become candidate
sim layouts -- printed as Python tuples ready for a seed module.
"""

import json
from pathlib import Path

from tankpit_bot.analysis.scan import decode_session_frames, load_capture_session
from tankpit_bot.protocol import decode_message

# Migrated 2026-08-06 onto tankpit_bot.analysis.scan (the typed
# capture-scan owner) - the private load/XOR/frame-walk pipeline is
# deleted; results reproduce exactly. load_capture_session is used
# directly (rather than scan_session) because the sim-capture filter
# needs the magic's content, which the scan result does not carry.

layouts: list[tuple[str, tuple[int, int], list[tuple[int, int, int, int, int]]]] = []

for path in sorted(Path("runs").glob("*/*.capture_session.json")):
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not session.get("magic") or "simmagic" in str(session.get("magic")):
        continue
    self_id: int | None = None
    first_map = None
    frames = decode_session_frames(load_capture_session(path))
    for frame in sorted(frames, key=lambda f: f["timestamp_ms"]):
        if frame["direction"] != "received":
            continue
        try:
            dm = decode_message(frame["msg_type"], frame["body"])
        except Exception:
            continue
        if dm["msg_type"] == 0x21 and self_id is None:
            self_id = dm["tank_id"]
        elif dm["msg_type"] == 0x4C and first_map is None:
            first_map = dm
        if first_map is not None and self_id is not None:
            break
    if first_map is None or self_id is None:
        continue
    bots = [
        (t["tank_id"], t["team"], t["rank"], t["x"], t["y"])
        for t in first_map["tanks"]
        if 500 <= t["tank_id"] <= 599 and not (t["x"] == 0 and t["y"] == 0)
    ]
    self_entries = [t for t in first_map["tanks"] if t["tank_id"] == self_id]
    if not self_entries or len(bots) < 20:
        continue
    spawn = (self_entries[0]["x"], self_entries[0]["y"])
    layouts.append((path.name, spawn, sorted(bots)))

print(f"sessions with rosters >= 20 bots: {len(layouts)}")
sizes = sorted({len(b) for _, _, b in layouts})
print("roster sizes seen:", sizes)
for name, spawn, bots in layouts[-4:]:
    print(f"\n# {name} spawn={spawn} bots={len(bots)}")
    teams = sorted({t for _, t, _, _, _ in bots})
    ranks = sorted({r for _, _, r, _, _ in bots})
    print(f"#  teams={teams} ranks={ranks}")
    print(bots[:12], "...")
