"""Mine WHY 0x4C map tank positions disagree with 0x3D wire positions.

Migrated 2026-08-06 onto ``tankpit_bot.analysis.scan`` (the typed
capture-scan owner) - the private load/XOR/frame-walk pipeline is
deleted; results reproduce exactly.

The displacement misclassification audit found that folding 0x4C map
entries into the 0x3D position table collapsed body attribution
1053->277 — the two channels measurably disagree about the same tank.
This miner pairs, per tank, every 0x4C map entry with that tank's
nearest-in-time 0x3D/0x28 wire fix inside a tight window and measures
the disagreement:

* exact-match rate — how often the channels agree tile-for-tile;
* delta histogram — the (map - wire) offsets, which name a lag
  (movement-sized scatter), a constant offset (off-by-N), or nothing;
* swapped-axes rate — map (x,y) equal to wire (y,x);
* scale hypotheses — map equal to wire doubled or halved.

Usage: python analysis_scripts/mine_map_position_delta.py <capture|dir ...>
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.container.helpers import ContainerDecodeError
from tankpit_bot.protocol import decode_message, try_decode_plaintext_ack
from tankpit_bot.protocol.helpers import DecodeError
from tankpit_bot.sniffer.decoders import _is_text_route

PAIR_WINDOW_MS = 2_000


def mine(path: Path, agg: dict) -> None:
    # Transport is tankpit_bot.analysis.scan (the typed capture-scan
    # owner this script's private load/XOR/frame-walk was migrated onto,
    # 2026-08-06). A magicless or unframed capture is a typed skip.
    result = scan_session(path)
    if "reason" in result:
        agg["skipped_no_magic"] = int(agg.get("skipped_no_magic", 0)) + 1
        return
    # Per tank: list of (t, x, y) wire fixes, and (t, x, y) map entries.
    wire: dict[int, list[tuple[int, int, int]]] = {}
    mapped: dict[int, list[tuple[int, int, int]]] = {}
    for frame in result["frames"]:
        if frame["direction"] != "received":
            continue
        t = frame["timestamp_ms"]
        body = frame["body"]
        if not body or try_decode_plaintext_ack(frame["raw"]) is not None:
            continue
        if _is_text_route(frame["msg_type"], frame["raw"]):
            continue
        try:
            decoded = dict(decode_message(frame["msg_type"], body))
        except Exception:
            continue
        mt = decoded.get("msg_type")
        if mt in (0x28, 0x3D):
            tid, x, y = decoded.get("tank_id"), decoded.get("x"), decoded.get("y")
            if isinstance(tid, int) and isinstance(x, int) and isinstance(y, int):
                wire.setdefault(tid, []).append((t, x, y))
        elif mt == 0x4C:
            for entry in decoded.get("tanks", []):
                tid, x, y = entry.get("tank_id"), entry.get("x"), entry.get("y")
                if isinstance(tid, int) and isinstance(x, int) and isinstance(y, int):
                    mapped.setdefault(tid, []).append((t, x, y))

    for tid, map_fixes in mapped.items():
        wire_fixes = wire.get(tid, [])
        if not wire_fixes:
            agg["map_only_tanks"] += 1
            continue
        for mt_, mx, my in map_fixes:
            best = None
            best_dt = PAIR_WINDOW_MS + 1
            for wt, wx, wy in wire_fixes:
                dt = abs(mt_ - wt)
                if dt < best_dt:
                    best, best_dt = (wx, wy), dt
            if best is None or best_dt > PAIR_WINDOW_MS:
                continue
            wx, wy = best
            agg["pairs"] += 1
            bucket = "dt<250ms" if best_dt < 250 else ("dt<1s" if best_dt < 1000 else "dt<2s")
            if (mx, my) == (wx, wy):
                agg["exact"] += 1
                agg["exact_by_dt"][bucket] = agg["exact_by_dt"].get(bucket, 0) + 1
            elif (mx, my) == (wy, wx):
                agg["swapped"] += 1
            elif (mx, my) == (wx * 2, wy * 2) or (mx, my) == (wx // 2, wy // 2):
                agg["scaled"] += 1
            else:
                agg["delta_hist"][f"{mx - wx},{my - wy}"] += 1
                if abs(mx - wx) + abs(my - wy) == 2 and mx == wx:
                    agg["y_pm2_by_dt"][bucket] = agg["y_pm2_by_dt"].get(bucket, 0) + 1
            agg["pairs_by_dt"][bucket] = agg["pairs_by_dt"].get(bucket, 0) + 1


def main() -> int:
    paths: list[Path] = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.capture_session.json")))
        else:
            paths.append(path)
    agg: dict = {
        "sessions": 0,
        "pairs": 0,
        "exact": 0,
        "swapped": 0,
        "scaled": 0,
        "map_only_tanks": 0,
        "pairs_by_dt": {},
        "exact_by_dt": {},
        "y_pm2_by_dt": {},
        "delta_hist": Counter(),
    }
    for path in paths:
        try:
            mine(path, agg)
            agg["sessions"] += 1
        except (OSError, ValueError, KeyError, DecodeError, ContainerDecodeError) as error:
            print(f"SKIP {path.name}: {error}")
    agg["delta_hist"] = dict(agg["delta_hist"].most_common(20))
    print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
