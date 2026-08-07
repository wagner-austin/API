"""Archive sweep: are map fuel dots the team-exposed LARGE containers?

User model (2026-07-25): yellow map dots are large fuel containers
previously exposed by the player or a teammate; smaller fuel
containers never appear on the map, and plenty of containers stay
hidden until radared. Test against every archived capture by joining
0x4F radar container reveals (x, y, volume; 0xFFFF = equipment)
against the session's 0x4C fuel-dot atlas:

* P(coord is a dot | fuel volume band) -- the user model predicts a
  monotone rise with volume, with small volumes ~never dotted.
* became-dot dynamics: fuel reveals NOT in the atlas at reveal time
  whose coords appear in a LATER 0x4C of the same session -- direct
  evidence that exposure adds large containers to the shared map.
* equipment reveals joined against dots (expected ~0: dots are fuel).
"""

import json
from collections import Counter
from pathlib import Path

from tankpit_bot.analysis.scan import decode_session_frames, load_capture_session
from tankpit_bot.protocol import decode_message

# Migrated 2026-08-06 onto tankpit_bot.analysis.scan (the typed
# capture-scan owner) - the private load/XOR/frame-walk pipeline is
# deleted; results reproduce exactly. load_capture_session is used
# directly (rather than scan_session) because the sim-capture filter
# needs the magic's content, which the scan result does not carry.

BANDS = ((1, 499), (500, 509), (510, 549), (550, 599), (600, 799), (800, 1099), (1100, 32767))


def band_of(volume: int) -> str:
    """Return the label of the volume band containing ``volume``."""
    for lo, hi in BANDS:
        if lo <= volume <= hi:
            return f"{lo}-{hi}"
    return "0"


agg: Counter[str] = Counter()
dotted_by_band: Counter[str] = Counter()
revealed_by_band: Counter[str] = Counter()
became_dot_by_band: Counter[str] = Counter()
not_dotted_never_by_band: Counter[str] = Counter()

for path in sorted(Path("runs").glob("*/*.capture_session.json")):
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not session.get("magic") or "simmagic" in str(session.get("magic")):
        continue
    events: list[tuple[int, str, object]] = []
    for frame in sorted(decode_session_frames(load_capture_session(path)), key=lambda f: f["timestamp_ms"]):
        if frame["direction"] != "received":
            continue
        try:
            dm = decode_message(frame["msg_type"], frame["body"])
        except Exception:
            continue
        if dm["msg_type"] == 0x4C:
            events.append((frame["timestamp_ms"], "map", set(map(tuple, dm["fuel_dots"]))))
        elif dm["msg_type"] == 0x4F:
            for c in dm["containers"]:
                events.append((frame["timestamp_ms"], "reveal", (c["x"], c["y"], c["volume"])))

    atlases = [(ts, dots) for ts, kind, dots in events if kind == "map"]
    if not atlases:
        continue
    agg["sessions_with_maps"] += 1
    current: set[tuple[int, int]] = set()
    atlas_i = 0
    for ts, kind, payload in events:
        if kind == "map":
            current = payload  # type: ignore[assignment]
            atlas_i += 1
            continue
        if not current:
            continue  # no atlas yet this session
        x, y, volume = payload  # type: ignore[misc]
        coord = (x, y)
        if volume == -1:
            agg["equipment_reveals"] += 1
            if coord in current:
                agg["equipment_reveals_on_dot"] += 1
            continue
        if volume == 0:
            agg["fuel_reveals_zero_volume"] += 1
            continue
        band = band_of(volume)
        revealed_by_band[band] += 1
        if coord in current:
            dotted_by_band[band] += 1
        else:
            later = [dots for t2, dots in atlases if t2 > ts]
            if any(coord in dots for dots in later):
                became_dot_by_band[band] += 1
            elif later:
                not_dotted_never_by_band[band] += 1

print(dict(agg))
print(f"{'band':>10} {'reveals':>8} {'on-dot':>7} {'P(dot)':>7} {'became':>7} {'never':>6}")
for lo, hi in BANDS:
    band = f"{lo}-{hi}"
    n = revealed_by_band[band]
    if n == 0:
        continue
    d = dotted_by_band[band]
    print(
        f"{band:>10} {n:>8} {d:>7} {d / n:>7.2f} "
        f"{became_dot_by_band[band]:>7} {not_dotted_never_by_band[band]:>6}"
    )
