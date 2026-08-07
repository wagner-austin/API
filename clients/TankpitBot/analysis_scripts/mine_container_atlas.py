"""Mine the true per-room container atlas from the full capture archive.

Every real-wire capture (runs/bot, runs/sniff, runs/probe) is replayed
through the production decoders and every per-tile container statement
is extracted as one observation event:

* ``0x4F`` RadarScanResult containers — (x, y, v) with v: 0 = tile
  empty, -1 = equipment, >0 = fuel volume (per-tile authoritative).
* ``0x43`` CacheUpdate / ``0x5A`` ViewportUpdate cache bytes — same
  vocabulary, visible-layer authoritative for enumerated tiles.
* ``container_pickup`` (0x2E-tunneled) — (x, y, remaining_volume)
  drain receipts.

Events are keyed by room (the sent ``*`` SELECT + the ``+`` ROOM_LIST
room->field mapping) and stamped with the capture's absolute epoch
timestamps, so the aggregation can order observations ACROSS sessions
and answer the static-field question directly:

* agreement — tiles seen by multiple sessions at the same volume;
* drains — volume non-increasing over time (consumption only);
* refills — any volume INCREASE between observations (the
  regeneration law showing itself, with its dt distribution);
* type flips — fuel <-> equipment transitions (placement dynamics).

Outputs ``runs/analysis/container_observations.jsonl`` (every event)
and ``runs/analysis/container_atlas.json`` (per-room per-tile summary),
plus the printed report.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol import decode_message, try_decode_plaintext_ack
from tankpit_bot.sniffer.decoders import _is_text_route
from tankpit_bot.state.viewport_geometry import viewport_patch_world_coords

# Migrated 2026-08-06 onto tankpit_bot.analysis.scan (the typed
# capture-scan owner, direction-tagged frames) - the private
# load/XOR/frame-walk pipeline is deleted; results reproduce exactly.
# Sent-frame room SELECTs and text ROOM_LIST rows read frame["raw"]
# (never ciphered), as the production receive path does.

RUN_DIRS = ("runs/bot", "runs/sniff", "runs/probe")
OUT_DIR = Path("runs/analysis")


def _pickup_records(pickups):
    """Normalize pickup records (dicts or tuples) to (x, y, remaining)."""
    for record in pickups:
        if isinstance(record, dict):
            yield record["x"], record["y"], record["remaining_volume"]
        else:
            yield record[0], record[1], record[2]


def mine_capture(path: Path, events: list, errors: dict) -> dict:
    """Extract every container observation from one capture."""
    try:
        result = scan_session(path)
    except (OSError, ValueError):
        errors["unreadable"] += 1
        return {}
    if "reason" in result or not result["frames"]:
        errors["no_magic_or_messages"] += 1
        return {}
    stamp = path.name.split(".")[0]
    room = "?"
    room_images: dict[str, str] = {}
    counts = defaultdict(int)
    seq = 0

    def emit(t, x, y, v, src):
        # ``seq`` preserves intra-payload wire order: one tick's batch
        # shares a timestamp, and a (pre-pickup read, pickup remaining)
        # pair sorted by value instead of arrival order manufactures a
        # phantom volume increase (found 2026-08-01: the "within-session
        # refills" were dominated by exactly this collision).
        nonlocal seq
        seq += 1
        events.append((t, seq, room, room_images.get(room, "?"), x, y, v, src, stamp))
        counts[src] += 1

    for frame in sorted(result["frames"], key=lambda f: f["timestamp_ms"]):
        t = frame["timestamp_ms"]
        raw = frame["raw"]
        if frame["direction"] == "sent":
            if frame["msg_type"] == 0x2A:  # '*' room SELECT
                try:
                    room = raw[1:].decode("ascii")
                except UnicodeDecodeError:
                    pass
            continue
        if try_decode_plaintext_ack(raw) is not None:
            continue
        if _is_text_route(frame["msg_type"], raw):
            if frame["msg_type"] == 0x2B:  # '+' ROOM_LIST rows carry room->field
                parts = raw.decode("utf-8", errors="replace")[1:].split("|")
                if len(parts) >= 7 and parts[6].startswith("field"):
                    room_images[parts[0]] = parts[6]
            continue
        try:
            decoded = decode_message(frame["msg_type"], frame["body"])
        except Exception:
            errors["decode"] += 1
            continue
        msg_type = decoded.get("msg_type")
        if msg_type == 0x4F:
            for c in decoded["containers"]:
                emit(t, c["x"], c["y"], c["volume"], "r")
        elif msg_type == 0x43:
            for x, y, v in decoded["updates"]:
                emit(t, x, y, v, "c")
        elif msg_type == 0x5A:
            left, top = decoded["viewport_left"], decoded["viewport_top"]
            for e in decoded["entities"]:
                x, y = viewport_patch_world_coords(left, top, e["col"], e["row"])
                emit(t, x, y, e["cache_value"], "v")
        elif msg_type == "container_pickup":
            for x, y, remaining in _pickup_records(decoded["pickups"]):
                emit(t, x, y, remaining, "p")
    return {"stamp": stamp, "room": room, "field": room_images.get(room, "?"), **counts}


def analyze(events: list) -> None:
    """Aggregate observation timelines and print the atlas report."""
    by_tile = defaultdict(list)
    for t, seq, room, field, x, y, v, src, stamp in events:
        by_tile[(room, field, x, y)].append((t, seq, v, src, stamp))

    atlas = {}
    refills = []
    flips = []
    multi_session_tiles = 0
    agree_tiles = 0
    drain_consistent = 0
    for key, timeline in by_tile.items():
        timeline.sort()
        room, field, x, y = key
        stamps = {row[4] for row in timeline}
        fuel_reads = [v for _, _, v, _, _ in timeline if v > 0]
        kinds = {(-1 if v == -1 else 1) for _, _, v, _, _ in timeline if v != 0}
        atlas.setdefault(f"{room}|{field}", {})[f"{x},{y}"] = {
            "observations": len(timeline),
            "sessions": len(stamps),
            "first_ms": timeline[0][0],
            "last_ms": timeline[-1][0],
            "last_v": timeline[-1][2],
            "max_fuel": max(fuel_reads, default=0),
            "equipment_seen": -1 in {v for _, _, v, _, _ in timeline},
            # Any visible-layer sighting (0x5A patch / 0x43 cache) means
            # the container is EXPOSED on the map — the sim's ``dotted``
            # flag. Radar-only tiles are the hidden layer.
            "visible_seen": any(src in ("v", "c") for _, _, _, src, _ in timeline),
        }
        if len(kinds) > 1:
            flips.append((key, timeline))
        if len(stamps) >= 2:
            multi_session_tiles += 1
            if len(set(fuel_reads)) <= 1:
                agree_tiles += 1
            non_increasing = True
            prev = None
            for t, _seq, v, src, _ in timeline:
                if v < 0:
                    continue
                if prev is not None and v > prev:
                    non_increasing = False
                    refills.append((key, prev, v, t))
                prev = v
            if non_increasing:
                drain_consistent += 1

    rooms = sorted(atlas)
    print(f"observation events: {len(events)}")
    print(f"distinct container tiles: {len(by_tile)}")
    for room_key in rooms:
        tiles = atlas[room_key]
        multi = sum(1 for v in tiles.values() if v["sessions"] >= 2)
        equip = sum(1 for v in tiles.values() if v["equipment_seen"])
        print(
            f"  room {room_key}: {len(tiles)} tiles "
            f"({multi} seen in 2+ sessions, {equip} ever-equipment)"
        )
    print(f"tiles observed in 2+ sessions: {multi_session_tiles}")
    print(f"  ... with a single fuel volume across ALL reads: {agree_tiles}")
    print(f"  ... drain-consistent (volume never increases over time): {drain_consistent}")
    print(f"REFILL events (volume increased between observations): {len(refills)}")
    for (room, field, x, y), old, new, t in refills[:20]:
        print(f"    room {room} ({field}) tile ({x},{y}): {old} -> {new} at {t}")
    print(f"type-flip tiles (fuel <-> equipment): {len(flips)}")
    for (room, field, x, y), timeline in flips[:10]:
        flip_seq = [v for _, _, v, _, _ in timeline if v != 0][:12]
        print(f"    room {room} ({field}) tile ({x},{y}): {flip_seq}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "container_atlas.json").write_text(
        json.dumps(atlas, indent=1, sort_keys=True), encoding="utf-8"
    )
    with (OUT_DIR / "container_observations.jsonl").open("w", encoding="utf-8") as fh:
        for row in sorted(events):
            fh.write(json.dumps(row) + "\n")
    print(f"wrote {OUT_DIR / 'container_atlas.json'} and container_observations.jsonl")


def main() -> int:
    paths = []
    for run_dir in RUN_DIRS:
        paths.extend(sorted(Path(run_dir).glob("*.capture_session.json")))
    print(f"captures: {len(paths)}")
    events: list = []
    errors = defaultdict(int)
    for index, path in enumerate(paths):
        summary = mine_capture(path, events, errors)
        if summary and (index % 25 == 0 or index == len(paths) - 1):
            sys.stdout.write(f"  [{index + 1}/{len(paths)}] {summary}\n")
            sys.stdout.flush()
    print(f"errors: {dict(errors)}")
    analyze(events)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
