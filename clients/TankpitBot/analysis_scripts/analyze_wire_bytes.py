"""Exhaustive wire byte analysis across captures.

Processes capture sessions through the XOR decode + container decode pipeline
and dumps EVERY byte of EVERY container message type, correlated with tank
lifecycle events (kills, damage transitions, position changes).

Goal: crack the undecoded fields that carry alive/dead state, hit/miss
detail, and damage information the game client renders but the bot ignores.

Usage:
    poetry run python -m scripts.analyze_wire_bytes [capture.json ...]
    poetry run python -m scripts.analyze_wire_bytes runs/bot/bot-20260611-004505.capture_session.json

With no arguments, scans runs/bot/ for captures with combat activity.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from platform_core.logging import get_logger, setup_rich_logging

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.container.decoders import decode_container_message
from tankpit_bot.container.identification import identify_container_type

# Migrated 2026-08-06 onto tankpit_bot.analysis.scan (the typed
# capture-scan owner) - the private load/XOR/frame-walk pipeline is
# deleted. The original stripped one length prefix (data[2:]) and only
# looked at payloads whose FIRST frame was 0x2E; the per-frame walk is
# the correction. NOTE (measured 2026-08-06): the default runs/bot
# sweep prints "No data found" both before and after migration - the
# script's container-type keys predate the current container decoders,
# so it is preserved-but-dormant archaeology.

log = get_logger(__name__)


def _format_bytes(data: bytes) -> str:
    """Format bytes as spaced hex with decimal values below."""
    hex_str = " ".join(f"{b:02x}" for b in data)
    dec_str = " ".join(f"{b:3d}" for b in data)
    return f"hex=[{hex_str}]  dec=[{dec_str}]"


def _analyze_session(session_path: Path) -> dict[str, list[dict[str, object]]]:
    """Process one capture session and return all decoded container messages.

    Returns a dict keyed by container message type name, each value a list
    of dicts with timestamp, tank_id, all raw bytes, and decoded fields.
    """
    result = scan_session(session_path)
    if "reason" in result:
        log.warning("Skipping %s: %s", session_path.name, result["reason"])
        return {}

    results: dict[str, list[dict[str, object]]] = defaultdict(list)

    for frame in result["frames"]:
        if frame["direction"] != "received":
            continue
        if frame["msg_type"] != 0x2E:
            continue

        decoded_bytes = frame["body"]
        if len(decoded_bytes) < 1:
            continue

        container_type = identify_container_type(decoded_bytes)
        type_name = container_type.name

        try:
            decoded_msg = decode_container_message(decoded_bytes)
        except Exception as e:
            results["DECODE_ERROR"].append(
                {
                    "timestamp_ms": frame["timestamp_ms"],
                    "error": str(e),
                    "raw_hex": decoded_bytes.hex(),
                    "length": len(decoded_bytes),
                }
            )
            continue

        entry: dict[str, object] = {
            "timestamp_ms": frame["timestamp_ms"],
            "type": type_name,
            "length": len(decoded_bytes),
            "raw_hex": decoded_bytes.hex(),
            "raw_bytes": list(decoded_bytes),
        }

        # Extract ALL fields from every message type — nothing skipped
        match decoded_msg:
            case {
                "msg_type": "combat_hit",
                "direction": d,
                "attacker_id": aid,
                "combat_data": cd,
                "is_outgoing": out,
            }:
                weapon_byte = cd[-1] if len(cd) > 0 else -1
                entry.update(
                    {
                        "direction_byte": d,
                        "attacker_id": aid,
                        "is_outgoing": out,
                        "weapon_byte": weapon_byte,
                        "combat_data_hex": cd.hex(),
                        "combat_data_bytes": list(cd),
                        "cd_byte0": cd[0] if len(cd) > 0 else None,
                        "cd_byte1": cd[1] if len(cd) > 1 else None,
                        "cd_byte2": cd[2] if len(cd) > 2 else None,
                        "cd_byte3": cd[3] if len(cd) > 3 else None,
                        "cd_byte4": cd[4] if len(cd) > 4 else None,
                        "cd_byte5": cd[5] if len(cd) > 5 else None,
                        "cd_byte6": cd[6] if len(cd) > 6 else None,
                    }
                )

            case {
                "msg_type": "tank_update_compact"
                | "tank_update_extended"
                | "tank_update_full" as mt,
                "flags": f,
                "tank_id": tid,
                "status_data": sd,
            }:
                entry.update(
                    {
                        "sub_type": mt,
                        "flags": f,
                        "flags_hex": f"0x{f:02x}",
                        "flags_bits": f"{f:08b}",
                        "tank_id": tid,
                        "status_data_hex": sd.hex(),
                        "status_data_bytes": list(sd),
                    }
                )
                for i, b in enumerate(sd):
                    entry[f"sd_byte{i}"] = b

            case {
                "msg_type": "tank_status_short",
                "flags": f,
                "tank_id": tid,
                "damage_state": dmg,
                "rank": rank,
                "leaderboard_position": lb,
            }:
                entry.update(
                    {
                        "flags": f,
                        "flags_hex": f"0x{f:02x}",
                        "flags_bits": f"{f:08b}",
                        "tank_id": tid,
                        "damage_state": dmg,
                        "rank": rank,
                        "leaderboard_position": lb,
                        "extra_byte": decoded_bytes[8] if len(decoded_bytes) > 8 else None,
                    }
                )

            case {
                "msg_type": "position_update",
                "flags": f,
                "tank_id": tid,
                "x": x,
                "y": y,
                "extra_data": ed,
            }:
                entry.update(
                    {
                        "flags": f,
                        "flags_hex": f"0x{f:02x}",
                        "flags_bits": f"{f:08b}",
                        "tank_id": tid,
                        "x": x,
                        "y": y,
                        "extra_data_hex": ed.hex(),
                        "extra_data_bytes": list(ed),
                    }
                )
                for i, b in enumerate(ed):
                    entry[f"ed_byte{i}"] = b

            case {"msg_type": "deactivation_kill", "victim_id": vid, "killer_id": kid}:
                entry.update(
                    {
                        "victim_id": vid,
                        "killer_id": kid,
                    }
                )

            case {"msg_type": "deactivation_death", "flags": f, "killer_id": kid, "extra_data": ed}:
                entry.update(
                    {
                        "flags": f,
                        "killer_id": kid,
                        "extra_data_hex": ed.hex(),
                        "extra_data_bytes": list(ed),
                    }
                )

            case {
                "msg_type": "tank_registry",
                "flags": f,
                "tank_id": tid,
                "info_bytes": ib,
                "team": team,
                "tank_name": name,
                "military_rank": mr,
                "is_bot": bot,
                "is_container": ic,
            }:
                entry.update(
                    {
                        "flags": f,
                        "flags_hex": f"0x{f:02x}",
                        "flags_bits": f"{f:08b}",
                        "tank_id": tid,
                        "info_bytes_hex": ib.hex(),
                        "info_bytes_list": list(ib),
                        "team": team,
                        "tank_name": name,
                        "military_rank": mr,
                        "is_bot": bot,
                        "is_container": ic,
                    }
                )
                for i, b in enumerate(ib):
                    entry[f"ib_byte{i}"] = b

            case {
                "msg_type": "movement",
                "flags": f,
                "start_x": sx,
                "start_y": sy,
                "player_id": pid,
                "tank_id": tid,
                "waypoints": wp,
                "is_self": is_self,
            }:
                entry.update(
                    {
                        "flags": f,
                        "flags_hex": f"0x{f:02x}",
                        "flags_bits": f"{f:08b}",
                        "start_x": sx,
                        "start_y": sy,
                        "player_id": pid,
                        "tank_id": tid,
                        "waypoints": wp,
                        "waypoint_count": len(wp),
                        "is_self": is_self,
                    }
                )

            case {"msg_type": "tank_leave", "tank_id": tid, "flags": f, "extra_data": ed}:
                entry.update(
                    {
                        "flags": f,
                        "flags_hex": f"0x{f:02x}",
                        "tank_id": tid,
                        "extra_data_hex": ed.hex(),
                        "extra_data_bytes": list(ed),
                    }
                )

            case {"msg_type": "tank_status_sync", "sync_data": sd}:
                entry.update(
                    {
                        "sync_data_hex": sd.hex(),
                        "sync_data_bytes": list(sd),
                    }
                )

            case {"msg_type": "teleport_landed", "subtype": st}:
                entry.update({"subtype": st})

            case {
                "msg_type": "container_pickup",
                "x": x,
                "y": y,
                "volume": vol,
                "is_fuel": is_fuel,
            }:
                entry.update(
                    {
                        "x": x,
                        "y": y,
                        "volume": vol,
                        "is_fuel": is_fuel,
                    }
                )

            case {
                "msg_type": "radar_response",
                "container_count": cc,
                "containers": containers,
                "mines": mines,
            }:
                entry.update(
                    {
                        "container_count": cc,
                        "num_containers": len(containers),
                        "num_mines": len(mines),
                    }
                )

            case {
                "msg_type": "tip_notification",
                "subtype": st,
                "length": ln,
                "notification_data": nd,
            }:
                entry.update(
                    {
                        "subtype": st,
                        "data_length": ln,
                        "notification_hex": nd.hex(),
                    }
                )

            case {"msg_type": "unknown_container", "subtype": st, "length": ln, "data": d}:
                entry.update(
                    {
                        "subtype": st,
                        "data_length": ln,
                        "data_hex": d.hex(),
                        "data_bytes": list(d),
                    }
                )

            case _:
                entry["decoded_type"] = str(type(decoded_msg).__name__)

        results[type_name].append(entry)

    return dict(results)


def _print_combat_hit_analysis(hits: list[dict[str, object]]) -> None:
    """Print detailed combat_hit byte analysis."""
    print("\n" + "=" * 80)
    print("COMBAT_HIT — full combat_data byte dump")
    print("=" * 80)
    print(f"Total combat_hit messages: {len(hits)}")
    print()

    outgoing = [h for h in hits if h.get("is_outgoing")]
    incoming = [h for h in hits if not h.get("is_outgoing")]
    print(f"  Outgoing (our shots): {len(outgoing)}")
    print(f"  Incoming (enemy shots): {len(incoming)}")
    print()

    # Group by weapon_byte
    by_weapon: dict[int, list[dict[str, object]]] = defaultdict(list)
    for h in hits:
        wb = h.get("weapon_byte", -1)
        assert isinstance(wb, int)
        by_weapon[wb].append(h)

    weapon_names = {0: "MISS/single", 1: "DUAL_HIT", 2: "MISSILE_HIT", 3: "HOMING_HIT"}
    for wb in sorted(by_weapon.keys()):
        group = by_weapon[wb]
        name = weapon_names.get(wb, f"UNKNOWN_{wb}")
        print(f"  weapon_byte={wb} ({name}): {len(group)} messages")

    print()
    print("Per-byte value ranges across ALL combat_hit messages:")
    for byte_idx in range(7):
        key = f"cd_byte{byte_idx}"
        values = sorted(set(h.get(key) for h in hits if h.get(key) is not None))
        assert all(isinstance(v, int) for v in values)
        print(f"  combat_data[{byte_idx}]: unique_values={len(values)} range={values}")

    print()
    print("--- OUTGOING (our shots) detail ---")
    for h in outgoing:
        ts = h.get("timestamp_ms")
        aid = h.get("attacker_id")
        wb = h.get("weapon_byte")
        cd = h.get("combat_data_hex")
        cd_bytes = h.get("combat_data_bytes")
        print(f"  ts={ts} attacker={aid} weapon={wb} combat_data={cd} bytes={cd_bytes}")

    print()
    print("--- INCOMING (enemy shots at us) detail ---")
    for h in incoming[:20]:
        ts = h.get("timestamp_ms")
        aid = h.get("attacker_id")
        wb = h.get("weapon_byte")
        cd = h.get("combat_data_hex")
        cd_bytes = h.get("combat_data_bytes")
        print(f"  ts={ts} attacker={aid} weapon={wb} combat_data={cd} bytes={cd_bytes}")
    if len(incoming) > 20:
        print(f"  ... and {len(incoming) - 20} more")


def _print_tank_update_analysis(updates: list[dict[str, object]], type_name: str) -> None:
    """Print detailed tank_update status_data byte analysis."""
    print(f"\n{'=' * 80}")
    print(f"{type_name} — full status_data byte dump")
    print("=" * 80)
    print(f"Total messages: {len(updates)}")

    # How many bytes of status_data?
    sd_lengths = set()
    for u in updates:
        sd_bytes = u.get("status_data_bytes")
        if isinstance(sd_bytes, list):
            sd_lengths.add(len(sd_bytes))
    print(f"status_data lengths: {sorted(sd_lengths)}")

    # Group by tank_id
    by_tank: dict[int, list[dict[str, object]]] = defaultdict(list)
    for u in updates:
        tid = u.get("tank_id")
        assert isinstance(tid, int)
        by_tank[tid].append(u)

    print(f"Unique tank IDs: {sorted(by_tank.keys())}")
    print()

    # Per-byte analysis
    max_sd_len = max(sd_lengths) if sd_lengths else 0
    print("Per-byte value ranges across ALL messages:")
    for byte_idx in range(max_sd_len):
        key = f"sd_byte{byte_idx}"
        values = sorted(set(u.get(key) for u in updates if u.get(key) is not None))
        print(f"  status_data[{byte_idx}]: unique_values={len(values)} values={values[:30]}")

    print()
    print("Flags byte distribution:")
    flags_counts: dict[str, int] = defaultdict(int)
    for u in updates:
        fb = u.get("flags_bits")
        assert isinstance(fb, str)
        flags_counts[fb] += 1
    for fb in sorted(flags_counts.keys()):
        print(f"  flags={fb} (0x{int(fb, 2):02x}): {flags_counts[fb]}")

    print()
    print("--- Per-tank timeline (first 5 tanks) ---")
    for tid in sorted(by_tank.keys())[:5]:
        tank_msgs = by_tank[tid]
        print(f"\n  Tank {tid} ({len(tank_msgs)} messages):")
        for u in tank_msgs[:10]:
            ts = u.get("timestamp_ms")
            flags = u.get("flags_hex")
            sd = u.get("status_data_hex")
            sd_bytes = u.get("status_data_bytes")
            print(f"    ts={ts} flags={flags} sd={sd} bytes={sd_bytes}")
        if len(tank_msgs) > 10:
            print(f"    ... and {len(tank_msgs) - 10} more")


def _print_position_update_analysis(updates: list[dict[str, object]]) -> None:
    """Print detailed position_update extra_data byte analysis."""
    print(f"\n{'=' * 80}")
    print("POSITION_UPDATE — full extra_data byte dump")
    print("=" * 80)
    print(f"Total messages: {len(updates)}")

    # Group by tank_id
    by_tank: dict[int, list[dict[str, object]]] = defaultdict(list)
    for u in updates:
        tid = u.get("tank_id")
        assert isinstance(tid, int)
        by_tank[tid].append(u)

    print(f"Unique tank IDs: {sorted(by_tank.keys())}")

    # Per-byte analysis
    print()
    print("Per-byte value ranges across ALL position_update messages:")
    for byte_idx in range(7):
        key = f"ed_byte{byte_idx}"
        values = sorted(set(u.get(key) for u in updates if u.get(key) is not None))
        print(f"  extra_data[{byte_idx}]: unique_values={len(values)} values={values[:30]}")

    print()
    print("Flags byte distribution:")
    flags_counts: dict[str, int] = defaultdict(int)
    for u in updates:
        fb = u.get("flags_bits")
        assert isinstance(fb, str)
        flags_counts[fb] += 1
    for fb in sorted(flags_counts.keys()):
        print(f"  flags={fb} (0x{int(fb, 2):02x}): {flags_counts[fb]}")

    print()
    print("--- Per-tank timeline (first 5 tanks) ---")
    for tid in sorted(by_tank.keys())[:5]:
        tank_msgs = by_tank[tid]
        print(f"\n  Tank {tid} ({len(tank_msgs)} messages):")
        for u in tank_msgs[:10]:
            ts = u.get("timestamp_ms")
            x = u.get("x")
            y = u.get("y")
            flags = u.get("flags_hex")
            ed = u.get("extra_data_hex")
            ed_bytes = u.get("extra_data_bytes")
            print(f"    ts={ts} pos=({x},{y}) flags={flags} extra={ed} bytes={ed_bytes}")
        if len(tank_msgs) > 10:
            print(f"    ... and {len(tank_msgs) - 10} more")


def _print_tank_status_short_analysis(statuses: list[dict[str, object]]) -> None:
    """Print tank_status_short analysis with focus on damage transitions."""
    print(f"\n{'=' * 80}")
    print("TANK_STATUS_SHORT — damage state + flags + extra byte")
    print("=" * 80)
    print(f"Total messages: {len(statuses)}")

    # Group by tank_id
    by_tank: dict[int, list[dict[str, object]]] = defaultdict(list)
    for s in statuses:
        tid = s.get("tank_id")
        assert isinstance(tid, int)
        by_tank[tid].append(s)

    print(f"Unique tank IDs: {sorted(by_tank.keys())}")

    # Flags analysis
    print()
    print("Flags byte distribution:")
    flags_counts: dict[str, int] = defaultdict(int)
    for s in statuses:
        fb = s.get("flags_bits")
        assert isinstance(fb, str)
        flags_counts[fb] += 1
    for fb in sorted(flags_counts.keys()):
        print(f"  flags={fb} (0x{int(fb, 2):02x}): {flags_counts[fb]}")

    # Extra byte analysis
    print()
    extra_values = sorted(
        set(s.get("extra_byte") for s in statuses if s.get("extra_byte") is not None)
    )
    print(f"Extra byte (offset 8) unique values: {extra_values}")

    print()
    print("--- Per-tank damage timeline ---")
    for tid in sorted(by_tank.keys()):
        tank_msgs = by_tank[tid]
        print(f"\n  Tank {tid} ({len(tank_msgs)} messages):")
        for s in tank_msgs:
            ts = s.get("timestamp_ms")
            dmg = s.get("damage_state")
            rank = s.get("rank")
            flags = s.get("flags_hex")
            extra = s.get("extra_byte")
            lb = s.get("leaderboard_position")
            print(
                f"    ts={ts} damage={dmg} rank={rank} flags={flags} extra=0x{extra:02x} lb={lb}"
                if isinstance(extra, int)
                else f"    ts={ts} damage={dmg} rank={rank} flags={flags} extra={extra} lb={lb}"
            )


def _print_deactivation_analysis(
    kills: list[dict[str, object]], deaths: list[dict[str, object]]
) -> None:
    """Print deactivation message analysis."""
    print(f"\n{'=' * 80}")
    print("DEACTIVATION — kill and death events")
    print("=" * 80)
    print(f"Deactivation kills: {len(kills)}")
    print(f"Deactivation deaths: {len(deaths)}")

    if kills:
        print()
        print("--- Kill events ---")
        for k in kills:
            ts = k.get("timestamp_ms")
            vid = k.get("victim_id")
            kid = k.get("killer_id")
            raw = k.get("raw_hex")
            print(f"  ts={ts} victim={vid} killer={kid} raw={raw}")

    if deaths:
        print()
        print("--- Death events ---")
        for d in deaths:
            ts = d.get("timestamp_ms")
            kid = d.get("killer_id")
            flags = d.get("flags")
            ed = d.get("extra_data_hex")
            raw = d.get("raw_hex")
            print(
                f"  ts={ts} killer={kid} flags=0x{flags:02x} extra={ed} raw={raw}"
                if isinstance(flags, int)
                else f"  ts={ts} killer={kid} flags={flags} extra={ed} raw={raw}"
            )


def _print_movement_analysis(movements: list[dict[str, object]]) -> None:
    """Print movement message flags analysis."""
    print(f"\n{'=' * 80}")
    print("MOVEMENT — flags byte analysis")
    print("=" * 80)
    print(f"Total messages: {len(movements)}")

    flags_counts: dict[str, int] = defaultdict(int)
    for m in movements:
        fb = m.get("flags_bits")
        assert isinstance(fb, str)
        is_self = m.get("is_self")
        key = f"{fb} (0x{int(fb, 2):02x}) self={is_self}"
        flags_counts[key] += 1

    print()
    print("Flags distribution:")
    for fb in sorted(flags_counts.keys()):
        print(f"  {fb}: {flags_counts[fb]}")


def _print_tank_registry_analysis(registries: list[dict[str, object]]) -> None:
    """Print tank_registry message analysis."""
    print(f"\n{'=' * 80}")
    print("TANK_REGISTRY — full info_bytes dump")
    print("=" * 80)
    print(f"Total messages: {len(registries)}")

    tanks = [r for r in registries if not r.get("is_container")]
    containers = [r for r in registries if r.get("is_container")]
    print(f"  Tank entries: {len(tanks)}")
    print(f"  Container entries: {len(containers)}")

    # Flags analysis
    print()
    print("Flags distribution (tanks only):")
    flags_counts: dict[str, int] = defaultdict(int)
    for r in tanks:
        fb = r.get("flags_bits")
        assert isinstance(fb, str)
        flags_counts[fb] += 1
    for fb in sorted(flags_counts.keys()):
        print(f"  flags={fb} (0x{int(fb, 2):02x}): {flags_counts[fb]}")

    # Per-byte analysis of info_bytes for tanks
    if tanks:
        max_ib_len = max(
            len(r.get("info_bytes_list", []))
            for r in tanks
            if isinstance(r.get("info_bytes_list"), list)
        )
        print()
        print("Per-byte value ranges across tank registry info_bytes:")
        for byte_idx in range(max_ib_len):
            key = f"ib_byte{byte_idx}"
            values = sorted(set(r.get(key) for r in tanks if r.get(key) is not None))
            print(f"  info_bytes[{byte_idx}]: unique_values={len(values)} values={values[:30]}")

    print()
    print("--- Tank entries detail (first 20) ---")
    for r in tanks[:20]:
        ts = r.get("timestamp_ms")
        tid = r.get("tank_id")
        name = r.get("tank_name")
        team = r.get("team")
        rank = r.get("military_rank")
        flags = r.get("flags_hex")
        ib = r.get("info_bytes_hex")
        ib_list = r.get("info_bytes_list")
        print(
            f"  ts={ts} id={tid} name={name} team={team} rank={rank} flags={flags} info={ib} bytes={ib_list}"
        )


def _print_unknown_analysis(unknowns: list[dict[str, object]]) -> None:
    """Print unknown container messages."""
    print(f"\n{'=' * 80}")
    print("UNKNOWN — unidentified container messages")
    print("=" * 80)
    print(f"Total: {len(unknowns)}")

    for u in unknowns:
        ts = u.get("timestamp_ms")
        length = u.get("length")
        raw = u.get("raw_hex")
        raw_bytes = u.get("raw_bytes")
        print(f"  ts={ts} len={length} raw={raw} bytes={raw_bytes}")


def _print_summary(results: dict[str, list[dict[str, object]]]) -> None:
    """Print overall message type summary."""
    print("\n" + "=" * 80)
    print("MESSAGE TYPE SUMMARY")
    print("=" * 80)
    total = sum(len(v) for v in results.values())
    print(f"Total container messages decoded: {total}")
    print()
    for type_name in sorted(results.keys()):
        count = len(results[type_name])
        print(f"  {type_name}: {count}")


def main() -> None:
    """Run exhaustive wire byte analysis."""
    setup_rich_logging(level="WARNING")

    if len(sys.argv) > 1:
        paths = [Path(p) for p in sys.argv[1:]]
    else:
        bot_dir = Path("runs/bot")
        if not bot_dir.exists():
            print("No runs/bot/ directory and no arguments given")
            sys.exit(1)
        paths = sorted(bot_dir.glob("*.capture_session.json"))
        print(f"Found {len(paths)} capture files in runs/bot/")
        print("Scanning for captures with combat activity...")

    all_results: dict[str, list[dict[str, object]]] = defaultdict(list)

    for path in paths:
        if not path.exists():
            print(f"File not found: {path}")
            continue

        try:
            results = _analyze_session(path)
        except Exception as e:
            print(f"Error processing {path.name}: {e}")
            continue

        has_combat = bool(
            results.get("COMBAT_HIT")
            or results.get("DEACTIVATION_KILL")
            or results.get("DEACTIVATION_DEATH")
        )

        if len(sys.argv) <= 1 and not has_combat:
            continue

        total = sum(len(v) for v in results.values())
        combat_count = len(results.get("COMBAT_HIT", []))
        kill_count = len(results.get("DEACTIVATION_KILL", []))
        print(f"\n{'-' * 60}")
        print(f"SESSION: {path.name}")
        print(f"  Total container messages: {total}")
        print(f"  Combat hits: {combat_count}")
        print(f"  Deactivation kills: {kill_count}")

        for type_name, entries in results.items():
            all_results[type_name].extend(entries)

    if not all_results:
        print("No data found.")
        sys.exit(0)

    print(f"\n{'=' * 80}")
    print("AGGREGATE ANALYSIS ACROSS ALL SESSIONS")
    print(f"{'=' * 80}")

    _print_summary(all_results)

    if all_results.get("COMBAT_HIT"):
        _print_combat_hit_analysis(all_results["COMBAT_HIT"])

    for tank_type in ["TANK_UPDATE_COMPACT", "TANK_UPDATE_EXTENDED", "TANK_UPDATE_FULL"]:
        if all_results.get(tank_type):
            _print_tank_update_analysis(all_results[tank_type], tank_type)

    if all_results.get("TANK_STATUS_SHORT"):
        _print_tank_status_short_analysis(all_results["TANK_STATUS_SHORT"])

    if all_results.get("POSITION_UPDATE"):
        _print_position_update_analysis(all_results["POSITION_UPDATE"])

    _print_deactivation_analysis(
        all_results.get("DEACTIVATION_KILL", []),
        all_results.get("DEACTIVATION_DEATH", []),
    )

    if all_results.get("MOVEMENT"):
        _print_movement_analysis(all_results["MOVEMENT"])

    if all_results.get("TANK_REGISTRY"):
        _print_tank_registry_analysis(all_results["TANK_REGISTRY"])

    if all_results.get("TANK_LEAVE"):
        print(f"\n{'=' * 80}")
        print("TANK_LEAVE")
        print("=" * 80)
        for tl in all_results["TANK_LEAVE"]:
            ts = tl.get("timestamp_ms")
            tid = tl.get("tank_id")
            flags = tl.get("flags_hex")
            ed = tl.get("extra_data_hex")
            raw = tl.get("raw_hex")
            print(f"  ts={ts} tank={tid} flags={flags} extra={ed} raw={raw}")

    if all_results.get("TANK_STATUS_SYNC"):
        print(f"\n{'=' * 80}")
        print(f"TANK_STATUS_SYNC — {len(all_results['TANK_STATUS_SYNC'])} messages")
        print("=" * 80)
        sync_data_values: dict[str, int] = defaultdict(int)
        for s in all_results["TANK_STATUS_SYNC"]:
            sd = s.get("sync_data_hex")
            assert isinstance(sd, str)
            sync_data_values[sd] += 1
        for sd in sorted(sync_data_values.keys()):
            print(f"  sync_data={sd}: {sync_data_values[sd]}")

    if all_results.get("UNKNOWN"):
        _print_unknown_analysis(all_results["UNKNOWN"])

    # Dump full JSON for offline analysis
    json_path = Path("wire_byte_analysis.json")
    serializable = {}
    for type_name, entries in all_results.items():
        clean_entries = []
        for e in entries:
            clean = {}
            for k, v in e.items():
                if isinstance(v, bytes):
                    clean[k] = v.hex()
                else:
                    clean[k] = v
            clean_entries.append(clean)
        serializable[type_name] = clean_entries
    json_path.write_text(json.dumps(serializable, indent=2))
    print(f"\nFull data dumped to {json_path} ({json_path.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()


__all__ = [
    "main",
]
