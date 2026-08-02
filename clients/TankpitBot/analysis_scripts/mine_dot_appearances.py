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

from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol import decode_message
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode

agg: Counter[str] = Counter()
unpreceded_volumes: Counter[str] = Counter()

for path in sorted(Path("runs").glob("*/*.capture_session.json")):
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not session.get("magic") or "simmagic" in str(session.get("magic")):
        continue
    reset_xor_state()
    build_global_xor_table(session["magic"])
    events: list[tuple[int, str, object]] = []
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
            if dm["msg_type"] == 0x4C:
                events.append((m["timestamp_ms"], "map", set(map(tuple, dm["fuel_dots"]))))
            elif dm["msg_type"] == 0x4F:
                for c in dm["containers"]:
                    if c["volume"] > 0:
                        events.append((m["timestamp_ms"], "reveal", (c["x"], c["y"], c["volume"])))
            elif dm["msg_type"] == 0x5A:
                left, top = dm["viewport_left"], dm["viewport_top"]
                for e in dm["entities"]:
                    if e["cache_value"] > 0:
                        events.append(
                            (
                                m["timestamp_ms"],
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
