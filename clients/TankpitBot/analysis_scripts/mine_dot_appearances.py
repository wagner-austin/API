"""Archive sweep: is every new map dot preceded by our own exposure?

Converse of ``mine_map_dot_semantics.py``. The 2026-07-22 mining read
within-session 0x4C atlas diffs as SPAWNS (~1/min). The user model
(2026-07-25) says dots are EXPOSURES of pre-existing >=500-volume fuel
containers. Discriminator: diff consecutive atlases per session; for
each appearing coordinate, look for any prior same-session reveal at
that coordinate -- an 0x4F radar container or an 0x5A viewport entity
with fuel cache_value. Appearances with NO prior reveal in our wire
are the residual (teammate exposure -- invisible to our capture -- or
true spawning).
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

agg: Counter[str] = Counter()
unpreceded_volumes: Counter[str] = Counter()

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
        ts = frame["timestamp_ms"]
        try:
            dm = decode_message(frame["msg_type"], frame["body"])
        except Exception:
            continue
        if dm["msg_type"] == 0x4C:
            events.append((ts, "map", set(map(tuple, dm["fuel_dots"]))))
        elif dm["msg_type"] == 0x4F:
            for c in dm["containers"]:
                if c["volume"] > 0:
                    events.append((ts, "reveal", (c["x"], c["y"], c["volume"])))
        elif dm["msg_type"] == 0x5A:
            left, top = dm["viewport_left"], dm["viewport_top"]
            for e in dm["entities"]:
                if e["cache_value"] > 0:
                    events.append(
                        (
                            ts,
                            "reveal",
                            (left + e["col"], top + e["row"], e["cache_value"]),
                        )
                    )

    prev_atlas: set[tuple[int, int]] | None = None
    revealed: dict[tuple[int, int], int] = {}
    for ts, kind, payload in events:
        if kind == "reveal":
            x, y, volume = payload  # type: ignore[misc]
            revealed[(x, y)] = max(volume, revealed.get((x, y), 0))
            continue
        atlas = payload  # type: ignore[assignment]
        if prev_atlas is not None:
            for coord in atlas - prev_atlas:  # type: ignore[operator]
                agg["appearances"] += 1
                volume = revealed.get(coord)
                if volume is None:
                    agg["unpreceded"] += 1
                elif volume >= 500:
                    agg["preceded_by_large_reveal"] += 1
                else:
                    agg["preceded_by_small_reveal"] += 1
                    unpreceded_volumes[str(volume)] += 1
        prev_atlas = atlas  # type: ignore[assignment]

print(dict(agg))
print("small-reveal volumes at appearing dots:", dict(unpreceded_volumes))
