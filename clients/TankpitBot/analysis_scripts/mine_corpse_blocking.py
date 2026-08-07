"""Does a corpse block walking? Mine the archive before probing live.

The corpse LIFETIME is proven (kill -> 0x58 = 22 s, the green
corpse-window shadow law). Unproven: whether the body BLOCKS walks
during that window — the assumption `state/occupancy.py` encodes and
F6 open question 2 rests on. Kills drop no loot (user contract
2026-08-04) — but the bot restocks from the current viewport after
every kill, and those collection routes cross corpse tiles often
enough that the archive should already hold the answer.

For every 0x41 kill in a capture: fix the corpse tile (the victim's
last wire-stated position before the kill), find its 0x58, then
classify every SELF 0x47 walk echo around the window:

* a path STEP ONTO the corpse tile BEFORE the 0x58 — disproof
  (corpses don't block);
* a walk that stops cardinally adjacent to the corpse tile with a
  code-1 receipt in-window — proof (corpses block like tanks);
* a step onto the tile AFTER the 0x58 — passability resumes.

Usage: ``python analysis_scripts/mine_corpse_blocking.py <capture ...>``
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol import decode_message, try_decode_plaintext_ack
from tankpit_bot.sniffer.decoders import _is_text_route

# Migrated 2026-08-06 onto tankpit_bot.analysis.scan (the typed
# capture-scan owner) - the private load/XOR/frame-walk pipeline is
# deleted; results reproduce exactly. Pre-cipher discriminators (ack,
# text route) read frame["raw"], as the production receive path does.

_STEP_DELTAS = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0)}


def _decode_all(path: Path) -> list[tuple[int, dict]]:
    result = scan_session(path)
    if "reason" in result:
        return []
    out: list[tuple[int, dict]] = []
    for frame in sorted(result["frames"], key=lambda f: f["timestamp_ms"]):
        if frame["direction"] != "received":
            continue
        if try_decode_plaintext_ack(frame["raw"]) is not None:
            continue
        if _is_text_route(frame["msg_type"], frame["raw"]):
            continue
        try:
            decoded = dict(decode_message(frame["msg_type"], frame["body"]))
        except Exception:
            continue
        out.append((frame["timestamp_ms"], decoded))
    return out


def _path_tiles(start_x: int, start_y: int, path: str) -> list[tuple[int, int]]:
    """Every tile a walk echo enters, in order."""
    tiles = []
    x, y = start_x, start_y
    for step in path:
        dx, dy = _STEP_DELTAS[step]
        x, y = x + dx, y + dy
        tiles.append((x, y))
    return tiles


def _hms(t: int) -> str:
    return datetime.datetime.fromtimestamp(t / 1000).strftime("%H:%M:%S")


def mine(path: Path) -> tuple[int, int, int]:
    decoded = _decode_all(path)
    self_id = next(
        m["tank_id"]
        for _, m in decoded
        if m.get("msg_type") == 0x21 and isinstance(m.get("tank_id"), int)
    )

    # Victim position tracking: last wire-stated (x, y) per tank.
    # 0x28/0x3D carry x,y directly; a 0x47 echo's final tile is
    # start+path.
    print(f"\n===== {path.name}: self={self_id} =====")
    on_before = on_after = adjacent_stop = 0
    position: dict[int, tuple[int, int]] = {}
    kills: list[tuple[int, int, tuple[int, int]]] = []  # (t0, victim, tile)
    removes: dict[int, int] = {}
    for t, msg in decoded:
        msg_type = msg.get("msg_type")
        tank_id = msg.get("tank_id")
        if msg_type in (0x28, 0x3D) and isinstance(tank_id, int) and isinstance(msg.get("x"), int):
            position[tank_id] = (msg["x"], msg["y"])
        elif msg_type == 0x47 and isinstance(tank_id, int) and isinstance(msg.get("path"), str):
            tiles = _path_tiles(msg["start_x"], msg["start_y"], msg["path"])
            if tiles:
                position[tank_id] = tiles[-1]
        elif msg_type == 0x41 and isinstance(msg.get("victim_id"), int):
            victim = msg["victim_id"]
            if victim != self_id and victim in position:
                kills.append((t, victim, position[victim]))
        elif msg_type == 0x58 and isinstance(tank_id, int):
            removes.setdefault(tank_id, t)

    for t0, victim, corpse in kills:
        t_remove = next(
            (t for t, m in decoded if m.get("msg_type") == 0x58 and m.get("tank_id") == victim and t > t0),
            t0 + 22_000,
        )
        for t, msg in decoded:
            if msg.get("msg_type") != 0x47 or msg.get("tank_id") != self_id:
                continue
            if not isinstance(msg.get("path"), str) or not msg["path"]:
                continue
            if not (t0 <= t <= t_remove + 15_000):
                continue
            tiles = _path_tiles(msg["start_x"], msg["start_y"], msg["path"])
            if corpse in tiles:
                if t <= t_remove:
                    on_before += 1
                    print(
                        f"  !! {_hms(t)} self walked ONTO corpse tile {corpse} "
                        f"of victim {victim} at +{(t - t0) / 1000:.1f}s "
                        f"(0x58 at +{(t_remove - t0) / 1000:.1f}s)  <-- DISPROOF"
                    )
                else:
                    on_after += 1
                    print(
                        f"     {_hms(t)} self crossed {corpse} at "
                        f"+{(t - t0) / 1000:.1f}s, {(t - t_remove) / 1000:.1f}s "
                        f"AFTER the 0x58 (cleared)"
                    )
            elif t <= t_remove and tiles:
                fx, fy = tiles[-1]
                if abs(fx - corpse[0]) + abs(fy - corpse[1]) == 1:
                    adjacent_stop += 1
                    print(
                        f"     {_hms(t)} self stopped ADJACENT to corpse "
                        f"{corpse} at +{(t - t0) / 1000:.1f}s "
                        f"(final ({fx},{fy}))"
                    )
    print(
        f"summary {path.name}: {len(kills)} kills, "
        f"{on_before} on-tile BEFORE 0x58, {on_after} on-tile after, "
        f"{adjacent_stop} adjacent stops in-window"
    )
    return on_before, on_after, adjacent_stop


def main() -> int:
    totals = (0, 0, 0)
    for arg in sys.argv[1:]:
        result = mine(Path(arg))
        totals = tuple(a + b for a, b in zip(totals, result))
    print(
        f"\nTOTAL: on-tile before 0x58 = {totals[0]} (disproofs), "
        f"on-tile after = {totals[1]}, adjacent stops = {totals[2]}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
