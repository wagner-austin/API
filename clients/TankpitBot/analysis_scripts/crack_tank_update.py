"""Crack the meaning of every byte in container TankUpdate* status_data.

We have three container-only message lengths (10 / 14 / 15 bytes inside a
0x2E envelope) that the Python decoder labels TankUpdateCompact /
TankUpdateExtended / TankUpdateFull. JS has no length-based dispatcher
for these (V['.'] = Og is the only 0x2E handler), so the labels and
per-byte field names are Python-side inferences.

This script grinds production captures to ground-truth those bytes:

  1. For every TankUpdate* body, extract (timestamp, tank_id,
     status_data bytes).
  2. For every 0x3D MovementResponse body, extract (timestamp,
     tank_id, x, y, direction, damage, rank, lb_score) per JS Mg.h.
  3. For every TankUpdate, find the *nearest* MovementResponse for
     the same tank_id within +/- 2 seconds and emit a side-by-side
     comparison.
  4. Across the whole corpus, count how often each status_data byte
     matches each MovementResponse field. The fields that have a high
     hit-rate are the real meaning of those bytes.

Usage::

    poetry run python -m analysis_scripts.crack_tank_update [capture.json ...]

With no arguments, scans runs/bot/*.capture_session.json.
"""

from __future__ import annotations

import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol.decoders.movement import decode_movement_response
from tankpit_bot.protocol.decoders.session_events import decode_build_pickup, decode_statistics
from tankpit_bot.protocol.decoders.tank import (
    decode_0x2e_message,
    decode_tank_entry,
)

# Migrated 2026-08-06 onto tankpit_bot.analysis.scan (the typed
# capture-scan owner) - the private base64/XOR-table/frame-walk
# pipeline is deleted; results reproduce exactly. Same day: the
# imports were repaired (protocol.decoders.misc no longer exists -
# decode_statistics / decode_build_pickup live in session_events).


@dataclass(frozen=True)
class TankUpdateSample:
    """One container TankUpdate body."""

    timestamp_ms: int
    body_len: int
    tank_id: int
    status_data: bytes
    full_body: bytes


@dataclass(frozen=True)
class MovementResponseSample:
    """One 0x3D MovementResponse body (Mg.h field layout)."""

    timestamp_ms: int
    tank_id: int
    team: int
    x: int
    y: int
    direction: int
    damage: int
    rank: int
    lb_score: int
    carrying: int


def _parse_movement_response(decoded: bytes) -> MovementResponseSample | None:
    """Decode an 0x3D-subtype MovementResponse body via the canonical decoder.

    JS Mg.h is reached only through the 0x2E container envelope -- the
    top-level wire byte 0x3D '=' is a TEXT JoinConfirm and never a
    binary MovementResponse. So ``decoded`` here is the *inner* payload
    (the 0x2E envelope minus subtype byte) that the runtime would hand
    decode_movement_response.
    """
    if len(decoded) < 11:
        return None
    try:
        result = decode_movement_response(decoded)
    except Exception:
        return None
    return MovementResponseSample(
        timestamp_ms=0,
        tank_id=result["tank_id"],
        team=result["team"],
        x=result["x"],
        y=result["y"],
        direction=result["direction"],
        damage=result["damage_state"],
        rank=result["rank"],
        lb_score=result["lb_score"],
        carrying=result["carrying"],
    )


def _parse_tank_update(decoded: bytes) -> TankUpdateSample | None:
    """Recognize a container TankUpdate* by length, parse common header."""
    if len(decoded) not in (10, 14, 15):
        return None
    tank_id = decoded[2] | (decoded[3] << 8)
    status_data = decoded[4:]
    return TankUpdateSample(
        timestamp_ms=0,
        body_len=len(decoded),
        tank_id=tank_id,
        status_data=status_data,
        full_body=decoded,
    )


def _count_normal_tank_entries(path: Path) -> int:
    """Count 0x2E bodies that currently route to TankEntry (subtype 0x28, inner >= 10).

    This is the comparison baseline for the 10-byte / 0x28 hypothesis:
    if regular TankEntry traffic is common (say, hundreds of samples)
    and the misrouted 10-byte / 0x28 count is 1, then we are looking
    at a real but very rare edge case (e.g. a tank join with empty
    name field). If the regular count is also near zero, the 10-byte
    case may be a pattern entirely unrelated to TankEntry.
    """
    result = scan_session(path)
    if "reason" in result:
        return 0
    count = 0
    for frame in result["frames"]:
        if frame["direction"] != "received" or frame["msg_type"] != 0x2E:
            continue
        decoded = frame["body"]
        if len(decoded) >= 11 and decoded[0] == 0x28:
            # Already qualifies for the tunneled TankEntry path
            count += 1
    return count


def _collect_samples(
    path: Path,
) -> tuple[list[TankUpdateSample], list[MovementResponseSample]]:
    """Walk one capture session and bucket every 0x2E TankUpdate and 0x3D."""
    result = scan_session(path)
    if "reason" in result:
        return [], []

    tank_updates: list[TankUpdateSample] = []
    movement_responses: list[MovementResponseSample] = []

    for frame in result["frames"]:
        if frame["direction"] != "received" or frame["msg_type"] != 0x2E:
            continue
        decoded = frame["body"]
        if len(decoded) < 1:
            continue
        timestamp = frame["timestamp_ms"]
        # Use the production subtype-first dispatcher so we route
        # tunneled subtypes (Movement at 0x47, ViewportUpdate at
        # 0x5A, MovementResponse at 0x3D, ...) the same way the
        # runtime does. Only bodies that fall through to the
        # length-based container path arrive as TankUpdate*.
        try:
            routed = decode_0x2e_message(decoded)
        except Exception:
            continue
        msg_type = routed.get("msg_type")
        if msg_type == 0x3D:
            movement_responses.append(
                MovementResponseSample(
                    timestamp_ms=timestamp,
                    tank_id=routed["tank_id"],
                    team=routed["team"],
                    x=routed["x"],
                    y=routed["y"],
                    direction=routed["direction"],
                    damage=routed["damage_state"],
                    rank=routed["rank"],
                    lb_score=routed["lb_score"],
                    carrying=routed["carrying"],
                )
            )
        elif msg_type in (
            "tank_update_compact",
            "tank_update_extended",
            "tank_update_full",
        ):
            tank_updates.append(
                TankUpdateSample(
                    timestamp_ms=timestamp,
                    body_len=len(decoded),
                    tank_id=routed["tank_id"],
                    status_data=routed["status_data"],
                    full_body=decoded,
                )
            )
    return tank_updates, movement_responses


def _nearest_movement(
    update: TankUpdateSample,
    movements: list[MovementResponseSample],
    window_ms: int = 2000,
) -> MovementResponseSample | None:
    """Return the MovementResponse for the same tank closest to update."""
    best: MovementResponseSample | None = None
    best_dt = window_ms + 1
    for m in movements:
        if m.tank_id != update.tank_id:
            continue
        dt = abs(m.timestamp_ms - update.timestamp_ms)
        if dt < best_dt:
            best_dt = dt
            best = m
    return best


def _candidate_fields(m: MovementResponseSample) -> dict[str, int]:
    """All ground-truth fields we might match a status_data byte against."""
    return {
        "x": m.x,
        "y": m.y,
        "direction": m.direction,
        "damage": m.damage,
        "rank": m.rank,
        "team": m.team,
        "carrying": m.carrying,
        "lb_score_hi": (m.lb_score >> 16) & 0xFF,
        "lb_score_mid": (m.lb_score >> 8) & 0xFF,
        "lb_score_lo": m.lb_score & 0xFF,
    }


def _byte_match_stats(
    pairs: list[tuple[TankUpdateSample, MovementResponseSample]],
    body_len: int,
    status_byte_idx: int,
) -> Counter[str]:
    """For one status_data byte position, count which field it matched."""
    hits: Counter[str] = Counter()
    total = 0
    for u, m in pairs:
        if u.body_len != body_len:
            continue
        if status_byte_idx >= len(u.status_data):
            continue
        total += 1
        byte = u.status_data[status_byte_idx]
        for field, value in _candidate_fields(m).items():
            if byte == value:
                hits[field] += 1
    hits["__total__"] = total
    return hits


def _print_report(
    tank_updates: list[TankUpdateSample],
    movements: list[MovementResponseSample],
) -> None:
    """Print the correlation summary across all sessions."""
    print(f"\nCollected {len(tank_updates)} TankUpdate bodies")
    print(f"Collected {len(movements)} 0x3D MovementResponse bodies")

    by_length: Counter[int] = Counter(u.body_len for u in tank_updates)
    for length, count in sorted(by_length.items()):
        print(f"  body_len={length}: {count}")

    pairs: list[tuple[TankUpdateSample, MovementResponseSample]] = []
    for u in tank_updates:
        m = _nearest_movement(u, movements)
        if m is not None:
            pairs.append((u, m))
    print(f"\nPaired {len(pairs)} TankUpdates with a 0x3D within +-2s")

    paired_lengths: Counter[int] = Counter(u.body_len for u, _ in pairs)
    for length in (10, 14, 15):
        n = paired_lengths[length]
        if n == 0:
            print(f"\n--- body_len={length}: NO pairs (no 0x3D for those tanks within +-2s)")
            continue
        print(f"\n=== body_len={length}: {n} paired samples")
        sd_len = length - 4
        for idx in range(sd_len):
            stats = _byte_match_stats(pairs, length, idx)
            total = stats.pop("__total__")
            print(f"  status_data[{idx}]  (total={total})")
            if not stats:
                print("    no field matches in any sample")
                continue
            for field, count in stats.most_common():
                pct = 100.0 * count / total
                marker = "  <-- LOCKED" if pct >= 95.0 else ""
                print(f"    {field:13s} {count:5d}/{total} ({pct:5.1f}%){marker}")


def main(argv: list[str]) -> int:
    if argv:
        paths = [Path(a) for a in argv]
    else:
        paths = sorted(
            list(Path("runs/bot").glob("*.capture_session.json"))
            + list(Path("runs/sniff").glob("*.capture_session.json"))
        )
    if not paths:
        print("No capture sessions found", file=sys.stderr)
        return 1

    all_updates: list[TankUpdateSample] = []
    all_movements: list[MovementResponseSample] = []
    normal_tank_entries = 0
    for path in paths:
        updates, movements = _collect_samples(path)
        all_updates.extend(updates)
        all_movements.extend(movements)
        normal_tank_entries += _count_normal_tank_entries(path)

    print(f"Processed {len(paths)} session(s)")
    print(
        f"Baseline: {normal_tank_entries} 0x2E bodies already qualify for "
        f"tunneled TankEntry (subtype 0x28, inner>=10)"
    )

    update_tids: Counter[int] = Counter(u.tank_id for u in all_updates)
    movement_tids: Counter[int] = Counter(m.tank_id for m in all_movements)
    print(
        f"\nDistinct tank_ids: updates={len(update_tids)} "
        f"movements={len(movement_tids)} overlap={len(set(update_tids) & set(movement_tids))}"
    )
    print("TankUpdate tank_id top 10:")
    for tid, n in update_tids.most_common(10):
        print(f"  tid={tid:5d} (0x{tid:04x}) -> {n}")
    print("MovementResponse tank_id top 10:")
    for tid, n in movement_tids.most_common(10):
        print(f"  tid={tid:5d} (0x{tid:04x}) -> {n}")

    print("\nSubtype byte distribution per TankUpdate length:")
    for target_len in (10, 14, 15):
        sub: Counter[int] = Counter(u.full_body[0] for u in all_updates if u.body_len == target_len)
        if not sub:
            print(f"  body_len={target_len}: (no samples)")
            continue
        print(f"  body_len={target_len}: {sum(sub.values())} samples")
        for byte, count in sub.most_common():
            char = chr(byte) if 32 <= byte < 127 else "."
            print(f"    byte0=0x{byte:02x} ('{char}') -> {count}")

    print("\n=== Verifying body_len=15 / subtype 0x56 as Statistics ===")
    stats_bodies = [u for u in all_updates if u.body_len == 15 and u.full_body[0] == 0x56]
    if stats_bodies:
        sane_count = 0
        insane: list[tuple[int, str, str]] = []
        for i, u in enumerate(stats_bodies):
            inner = u.full_body[1:]
            try:
                s = decode_statistics(inner)
            except Exception as e:
                insane.append((i, "decode_error", str(e)))
                continue
            # Sanity bounds: minutes 0-59, seconds 0-59, hours reasonable.
            problems: list[str] = []
            if not 0 <= s["playtime_minutes"] <= 59:
                problems.append(f"minutes={s['playtime_minutes']}")
            if not 0 <= s["playtime_seconds"] <= 59:
                problems.append(f"seconds={s['playtime_seconds']}")
            if problems:
                insane.append((i, ",".join(problems), inner.hex()))
            else:
                sane_count += 1
        total = len(stats_bodies)
        print(f"  Sane (minutes 0-59 AND seconds 0-59): {sane_count}/{total}")
        if insane:
            print(f"  Insane samples: {len(insane)} (showing first 5)")
            for idx, why, hexstr in insane[:5]:
                print(f"    #{idx}: {why}  hex={hexstr}")
        print("  First 5 decoded values:")
        for u in stats_bodies[:5]:
            try:
                s = decode_statistics(u.full_body[1:])
                print(
                    f"    hrs={s['playtime_hours']:5d} "
                    f"min={s['playtime_minutes']:2d} "
                    f"sec={s['playtime_seconds']:2d} "
                    f"destroyed={s['destroyed']:6d} "
                    f"deactivated={s['deactivated']:5d} "
                    f"score={s['score']:8d}"
                )
            except Exception as e:
                print(f"    DECODE ERROR: {e}")
    else:
        print("  No samples")

    print("\n=== Verifying body_len=10 / subtype 0x28 as TankEntry ===")
    entry_bodies = [u for u in all_updates if u.body_len == 10 and u.full_body[0] == 0x28]
    if entry_bodies:
        for i, u in enumerate(entry_bodies):
            inner = u.full_body[1:]
            try:
                e = decode_tank_entry(inner)
                print(
                    f"  #{i}: tid={e['tank_id']} team={e['team']} rank={e['rank']} "
                    f"dmg={e['damage_state']} score={e['score']} "
                    f"pos=({e['x']},{e['y']})  inner_hex={inner.hex()}"
                )
            except Exception as ex:
                print(f"  #{i}: DECODE ERROR: {ex}")
    else:
        print("  No samples")

    print("\n=== Verifying body_len=10 / subtype 0x42 as BuildPickup ===")
    build_bodies = [u for u in all_updates if u.body_len == 10 and u.full_body[0] == 0x42]
    if build_bodies:
        for i, u in enumerate(build_bodies):
            inner = u.full_body[1:]
            try:
                bp = decode_build_pickup(inner)
                # Sanity: source_x/source_y and drop_x/drop_y should be 0-255
                # tile coords; direction should be a small int.
                print(
                    f"  #{i}: tid={bp['tank_id']} "
                    f"src=({bp['source_x']},{bp['source_y']}) "
                    f"drop=({bp['drop_x']},{bp['drop_y']}) "
                    f"dir={bp['direction']} bridge={bp['is_bridge']} "
                    f"flag={bp['flag']}  inner_hex={inner.hex()}"
                )
            except Exception as ex:
                print(f"  #{i}: DECODE ERROR: {ex}")
    else:
        print("  No samples")

    print("\nSample TankUpdate full bodies (per length):")
    for target_len in (10, 14, 15):
        print(f"\n--- body_len={target_len} (first 10 raw bodies) ---")
        n = 0
        for u in all_updates:
            if u.body_len != target_len:
                continue
            tail_ascii = "".join(chr(b) if 32 <= b < 127 else "." for b in u.full_body)
            print(f"  hex=[{u.full_body.hex()}]  ascii=[{tail_ascii}]")
            n += 1
            if n >= 10:
                break
    print("\nSample MovementResponses:")
    for m in all_movements[:5]:
        print(
            f"  ts={m.timestamp_ms}  tid={m.tank_id} (0x{m.tank_id:04x})  "
            f"x={m.x} y={m.y} dir={m.direction} dmg={m.damage} rank={m.rank}"
        )

    _print_report(all_updates, all_movements)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
