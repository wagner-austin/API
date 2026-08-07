"""Byte-mine the ferry movement law from 0x4A terrain updates.

Ferries are wire TERRAIN (``TERRAIN_FERRY = 5`` / ``TERRAIN_FERRY_ROCK
= 7``), not tanks — the first cut of this miner searched 0x3D position
statements and found nothing, because a ferry's movement is announced
as terrain changes: a 0x4A triple restoring the old tile and one
painting the new. This version tracks the live ferry tile set through
every 0x4A, pairs leave/arrive triples into move steps, and classifies
each step by whether a tank stood on the moving tile:

* **ridden** — the known law (the ferry moves with its rider);
* **unridden** — autonomous drift, the unmined law this exists for:
  does an empty ferry move at all, on what cadence, with what step
  shape?

Usage: ``python analysis_scripts/mine_ferry_drift.py <capture ...>``
"""

from __future__ import annotations

import datetime
import sys
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol import decode_message, try_decode_plaintext_ack
from tankpit_bot.sniffer.decoders import _is_text_route
from tankpit_bot.state.viewport_geometry import viewport_patch_world_coords

# Migrated 2026-08-06 onto tankpit_bot.analysis.scan (the typed
# capture-scan owner) - the private load/XOR/frame-walk pipeline is
# deleted; results reproduce exactly. Pre-cipher discriminators (ack,
# text route) read frame["raw"], as the production receive path does.

FERRY_TYPES = {5, 7}
PAIR_WINDOW_MS = 1_500
# A ferry relocates in rider-move-sized LEGS, not single steps: the
# 2026-07-20 ride announced (223,195)->(226,196) — Manhattan 4 — as
# one atomic 0x4A pair (old tile restored to 0, new tile painted 5).
PAIR_MAX_TILES = 40

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


def _hms(t: int) -> str:
    return datetime.datetime.fromtimestamp(t / 1000).strftime("%H:%M:%S")


def _track_tile(
    t: int,
    tile: tuple[int, int],
    terrain_type: int,
    position: dict[int, tuple[int, int]],
    ferry_tiles: dict[tuple[int, int], int],
    departures: list[tuple[int, tuple[int, int]]],
    moves: list[tuple[int, tuple[int, int], tuple[int, int], bool, int]],
) -> int:
    """Feed one absolute (tile, terrain) statement into the tracker.

    Returns:
        1 for a NEW ferry-tile arrival (for the arrivals counter),
        else 0.
    """
    if terrain_type in FERRY_TYPES:
        if tile in ferry_tiles:
            return 0
        ferry_tiles[tile] = t
        for i, (dt_, old) in enumerate(departures):
            distance = abs(old[0] - tile[0]) + abs(old[1] - tile[1])
            if t - dt_ <= PAIR_WINDOW_MS and 0 < distance <= PAIR_MAX_TILES:
                ridden = any(pos == old for pos in position.values())
                moves.append((t, old, tile, ridden, t - dt_))
                departures.pop(i)
                break
        return 1
    if tile in ferry_tiles:
        del ferry_tiles[tile]
        departures.append((t, tile))
    return 0


def mine(path: Path) -> None:
    decoded = _decode_all(path)

    # Tank position tracking (0x28/0x3D direct, 0x47 echo finals) so a
    # moving ferry tile can be classified ridden/unridden.
    position: dict[int, tuple[int, int]] = {}
    ferry_tiles: dict[tuple[int, int], int] = {}  # tile -> arrival t
    departures: list[tuple[int, tuple[int, int]]] = []  # (t, tile) awaiting a pair
    moves: list[tuple[int, tuple[int, int], tuple[int, int], bool, int]] = []
    arrivals = 0

    for t, msg in decoded:
        msg_type = msg.get("msg_type")
        tank_id = msg.get("tank_id")
        if msg_type in (0x28, 0x3D) and isinstance(tank_id, int) and isinstance(msg.get("x"), int):
            position[tank_id] = (msg["x"], msg["y"])
        elif msg_type == 0x47 and isinstance(tank_id, int) and isinstance(msg.get("path"), str):
            x, y = msg["start_x"], msg["start_y"]
            for step in msg["path"]:
                dx, dy = _STEP_DELTAS[step]
                x, y = x + dx, y + dy
            position[tank_id] = (x, y)
        elif msg_type == 0x4A and isinstance(msg.get("updates"), list):
            for update in msg["updates"]:
                x, y, terrain_type = update
                arrivals += _track_tile(
                    t, (x, y), terrain_type, position, ferry_tiles, departures, moves
                )
        elif msg_type == 0x5A and isinstance(msg.get("entities"), list):
            vp_left, vp_top = msg["viewport_left"], msg["viewport_top"]
            for ent in msg["entities"]:
                x, y = viewport_patch_world_coords(vp_left, vp_top, ent["col"], ent["row"])
                arrivals += _track_tile(
                    t, (x, y), ent["terrain_type"], position, ferry_tiles, departures, moves
                )
        # Expire stale departures so distant pairs never form.
        departures = [(dt_, tile) for dt_, tile in departures if t - dt_ <= PAIR_WINDOW_MS]

    print(f"\n===== {path.name}: {arrivals} ferry-tile arrivals, {len(moves)} paired moves =====")
    unridden = [m for m in moves if not m[3]]
    ridden = [m for m in moves if m[3]]
    for t, old, new, _, gap in unridden[:20]:
        print(
            f"  UNRIDDEN {_hms(t)}: {old} -> {new} "
            f"delta=({new[0] - old[0]:+d},{new[1] - old[1]:+d}) pair_gap={gap}ms"
        )
    if ridden:
        gaps = sorted(m[4] for m in ridden)
        print(f"  ridden moves: {len(ridden)} (pair gap median {gaps[len(gaps) // 2]}ms)")
    print(f"summary: ridden={len(ridden)} unridden={len(unridden)}")


def main() -> int:
    for arg in sys.argv[1:]:
        mine(Path(arg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
