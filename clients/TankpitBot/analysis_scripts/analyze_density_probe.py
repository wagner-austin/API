"""Offline analysis of a density-probe capture: hidden-container density.

Reads ``density_probe.capture_session.json`` (or argv[1]): for every
0x4F radar response, attributes a 16x16 viewport window around the
nearest preceding self position and counts the DELTA reveals — the
hidden population of those 256 fresh tiles (0x4F never re-sends
already-visible entities). Separately counts exposed containers from
0x5A landing patches (cache_value > 0 fuel, -1 equipment) and
cross-checks >=500 reveals against the session's dot atlas.
"""

import json
import sys
from collections import Counter
from pathlib import Path

from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol import decode_message
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode

path = Path(sys.argv[1] if len(sys.argv) > 1 else "density_probe.capture_session.json")
session = json.loads(path.read_text(encoding="utf-8"))
reset_xor_state()
build_global_xor_table(session["magic"])

self_id: int | None = None
position: tuple[int, int] | None = None
atlas: set[tuple[int, int]] = set()
scans: list[dict[str, object]] = []
exposed_by_landing: list[int] = []
hidden_volumes: list[int] = []

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
        elif dm["msg_type"] == 0x3D and dm.get("tank_id") == self_id:
            position = (dm["x"], dm["y"])
        elif dm["msg_type"] == 0x4C:
            atlas |= set(map(tuple, dm["fuel_dots"]))
        elif dm["msg_type"] == 0x5A:
            count = sum(1 for e in dm["entities"] if e["cache_value"] != 0)
            if count:
                exposed_by_landing.append(count)
        elif dm["msg_type"] == 0x4F and dm["containers"]:
            fuel = [c for c in dm["containers"] if c["volume"] >= 0]
            equip = sum(1 for c in dm["containers"] if c["volume"] == -1)
            hidden_volumes.extend(c["volume"] for c in fuel if c["volume"] > 0)
            large_on_dot = sum(
                1 for c in fuel if c["volume"] >= 500 and (c["x"], c["y"]) in atlas
            )
            scans.append(
                {
                    "at": position,
                    "fuel": len(fuel),
                    "fuel_stocked": sum(1 for c in fuel if c["volume"] > 0),
                    "equip": equip,
                    "large": sum(1 for c in fuel if c["volume"] >= 500),
                    "large_on_dot": large_on_dot,
                }
            )

sites = [s for s in scans if s["fuel"] or s["equip"]]
print(f"scans with reveals: {len(scans)}")
for s in scans:
    print(" ", s)
total_tiles = 256 * len(scans)
if scans:
    total_fuel = sum(int(str(s["fuel_stocked"])) for s in scans)
    total_equip = sum(int(str(s["equip"])) for s in scans)
    print(f"\ntiles sampled: {total_tiles}")
    print(f"hidden stocked fuel / 256 tiles: {256 * total_fuel / total_tiles:.2f}")
    print(f"hidden equipment / 256 tiles: {256 * total_equip / total_tiles:.2f}")
bands = Counter()
for v in hidden_volumes:
    key = "<500" if v < 500 else ("500-799" if v < 800 else ("800-1099" if v < 1100 else "1100+"))
    bands[key] += 1
print("hidden volume bands:", dict(bands))
print("exposed containers per 0x5A landing:", exposed_by_landing)
print("atlas size:", len(atlas))
