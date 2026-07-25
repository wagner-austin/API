"""Archive sweep: extract real practice-room layouts for sim seeding.

For each archived capture, take the client id (first 0x21 TankInfo)
and the FIRST 0x4C map snapshot: every bot slot (tank_id 500-599)
with its team, rank, and position, plus the client's own map entry
(the join spawn). Sessions with a full-size roster become candidate
sim layouts -- printed as Python tuples ready for a seed module.
"""

import json
from pathlib import Path

from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol import decode_message
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode

layouts: list[tuple[str, tuple[int, int], list[tuple[int, int, int, int, int]]]] = []

for path in sorted(Path("runs").glob("*/*.capture_session.json")):
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not session.get("magic") or "simmagic" in str(session.get("magic")):
        continue
    reset_xor_state()
    build_global_xor_table(session["magic"])
    self_id: int | None = None
    first_map = None
    for m in sorted(session["messages"], key=lambda x: x["timestamp_ms"]):
        if m["direction"] != "received":
            continue
        data = decode_base64_safe(m["payload"])
        if not data:
            continue
        off = 0
        while off + 2 < len(data):
            ln = data[off] | (data[off + 1] << 8)
            off += 2
            if ln == 0 or off + ln > len(data):
                break
            body = data[off : off + ln]
            off += ln
            try:
                dm = decode_message(body[0], xor_decode(body))
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
