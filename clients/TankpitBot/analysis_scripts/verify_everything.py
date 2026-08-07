"""Verify and decode everything that's still unverified.

1. Cross-reference Supervisor kill timestamps vs inferred kills
2. Decode the 8 longer ActionDone messages
3. Inventory ALL unknown subtypes
4. Verify combat_data viewport offset math
5. Verify 0x2e[11] against FuelGain protocol messages
"""

import json
from collections import defaultdict
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session

# Migrated 2026-08-06 onto tankpit_bot.analysis.scan (the typed
# capture-scan owner). The original stripped one length prefix
# (data[2:]) and treated the remainder of the payload as a single
# frame; the per-frame walk is the correction, and the corpus
# reproduces the old output exactly.


def process_session(session_path: Path) -> dict[str, list[dict[str, object]]]:
    """Process ALL message types from a session."""
    result = scan_session(session_path)
    if "reason" in result:
        return {}

    results: dict[str, list[dict[str, object]]] = defaultdict(list)

    for frame in result["frames"]:
        if frame["direction"] != "received":
            continue
        msg_type = frame["msg_type"]
        ts = frame["timestamp_ms"]

        if msg_type == 0x2E:
            decoded = frame["body"]
            if len(decoded) < 1:
                continue

            # Supervisor tunneled inside 0x2E
            if decoded[0] == 0x52 and len(decoded) >= 4:
                results["supervisor"].append(
                    {
                        "timestamp_ms": ts,
                        "status": decoded[1],
                        "reserved": decoded[2],
                        "data": decoded[3],
                        "bytes": list(decoded),
                    }
                )
                continue

            # FuelGain tunneled (0x44)
            if decoded[0] == 0x44 and len(decoded) >= 3:
                fuel_total = decoded[1] | (decoded[2] << 8)
                if len(decoded) >= 4:
                    fuel_total = decoded[1] | (decoded[2] << 8) | (decoded[3] << 16)
                results["fuel_gain"].append(
                    {
                        "timestamp_ms": ts,
                        "fuel_total": fuel_total,
                        "bytes": list(decoded),
                    }
                )
                continue

            # ActionDone tunneled (0x54)
            if decoded[0] == 0x54:
                results["action_done"].append(
                    {
                        "timestamp_ms": ts,
                        "bytes": list(decoded),
                        "length": len(decoded),
                    }
                )
                continue

            # 0x3d and 0x2e self (already analyzed)
            if decoded[0] == 0x3D and len(decoded) == 13:
                b = list(decoded)
                tid = b[2] | (b[3] << 8)
                results["0x3d"].append(
                    {
                        "timestamp_ms": ts,
                        "tank_id": tid,
                        "x": b[4],
                        "y": b[5],
                        "direction": b[6],
                        "damage_state": b[7],
                        "rank": b[8],
                        "lb_high": b[10],
                        "rank_points": b[11],
                    }
                )
                continue

            if decoded[0] == 0x2E and len(decoded) == 13:
                b = list(decoded)
                tid = b[2] | (b[3] << 8)
                results["0x2e_self"].append(
                    {
                        "timestamp_ms": ts,
                        "tank_id": tid,
                        "damage_state": b[4],
                        "byte7": b[7],
                        "byte8": b[8],
                        "byte11": b[11],
                        "byte12": b[12],
                    }
                )
                continue

        else:
            decoded = frame["body"]
            # FuelGain standalone (0x44)
            if len(decoded) >= 3 and decoded[0] == 0x44:
                fuel_total = decoded[1] | (decoded[2] << 8)
                if len(decoded) >= 4:
                    fuel_total |= decoded[3] << 16
                results["fuel_gain"].append(
                    {
                        "timestamp_ms": ts,
                        "fuel_total": fuel_total,
                        "bytes": list(decoded),
                    }
                )

    return dict(results)


def main() -> None:
    from platform_core.logging import setup_rich_logging

    setup_rich_logging(level="WARNING")

    bot_dir = Path("runs/bot")
    paths = sorted(bot_dir.glob("*.capture_session.json"))

    all_results: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in paths:
        try:
            results = process_session(path)
        except Exception:
            continue
        for k, v in results.items():
            all_results[k].extend(v)

    # ================================================================
    # 1. Supervisor kill timestamps vs inferred kills from 0x3d
    # ================================================================
    print("=" * 80)
    print("VERIFY: Supervisor data=8 vs inferred kills from 0x3d")
    print("=" * 80)

    supervisors_kill = [s for s in all_results.get("supervisor", []) if s["data"] == 8]
    print(f"  Supervisor data=8 messages: {len(supervisors_kill)}")

    # Infer kills from 0x3d streams
    u3d = all_results.get("0x3d", [])
    u3d_by_tank: dict[int, list[dict[str, object]]] = defaultdict(list)
    for u in u3d:
        tid = u["tank_id"]
        assert isinstance(tid, int)
        if tid != 1301:
            u3d_by_tank[tid].append(u)

    for tid in u3d_by_tank:
        u3d_by_tank[tid].sort(key=lambda x: x["timestamp_ms"])

    kill_timestamps: list[tuple[int, int]] = []
    for tid, msgs in u3d_by_tank.items():
        for i in range(1, len(msgs)):
            prev_dmg = msgs[i - 1]["damage_state"]
            curr_dmg = msgs[i]["damage_state"]
            if curr_dmg == 0 and prev_dmg == 1:
                gap = msgs[i]["timestamp_ms"] - msgs[i - 1]["timestamp_ms"]
                if gap > 5000:
                    kill_timestamps.append((msgs[i - 1]["timestamp_ms"], tid))
            if curr_dmg == 1 and prev_dmg == 2:
                if (
                    i == len(msgs) - 1
                    or msgs[i + 1]["timestamp_ms"] - msgs[i]["timestamp_ms"] > 10000
                ):
                    kill_timestamps.append((msgs[i]["timestamp_ms"], tid))

    kill_timestamps.sort()
    print(f"  Inferred kills from 0x3d: {len(kill_timestamps)}")

    # Cross-reference
    matched_kills = 0
    unmatched_supervisors = 0
    for sv in supervisors_kill:
        sv_ts = sv["timestamp_ms"]
        assert isinstance(sv_ts, int)
        best_delta = 999999
        best_kill = None
        for kt, kid in kill_timestamps:
            delta = abs(sv_ts - kt)
            if delta < best_delta:
                best_delta = delta
                best_kill = (kt, kid)
        if best_delta < 5000:
            matched_kills += 1
            assert best_kill is not None
            print(
                f"  MATCH: supervisor ts={sv_ts} kill ts={best_kill[0]} tank={best_kill[1]} delta={best_delta}ms"
            )
        else:
            unmatched_supervisors += 1
            print(f"  UNMATCHED: supervisor ts={sv_ts} (nearest kill {best_delta}ms away)")

    print(f"\n  Matched: {matched_kills}/{len(supervisors_kill)}")
    print(f"  Unmatched supervisors: {unmatched_supervisors}")

    # ================================================================
    # 2. ALL supervisor message variants
    # ================================================================
    print()
    print("=" * 80)
    print("ALL SUPERVISOR VARIANTS")
    print("=" * 80)

    sv_variants: dict[tuple[int, int, int], int] = defaultdict(int)
    for s in all_results.get("supervisor", []):
        key = (s["status"], s["reserved"], s["data"])
        assert all(isinstance(v, int) for v in key)
        sv_variants[key] += 1

    for (status, reserved, data), count in sorted(sv_variants.items(), key=lambda x: -x[1]):
        labels = []
        if status == 1:
            labels.append("PROMO_ELIGIBLE")
        if data == 8:
            labels.append("PROMO_KILL")
        label = f" ({', '.join(labels)})" if labels else ""
        print(f"  status={status} reserved={reserved} data={data}: {count}{label}")

    # ================================================================
    # 3. Verify 0x2e[11] IS fuel low byte
    # Cross-reference with FuelGain messages
    # ================================================================
    print()
    print("=" * 80)
    print("VERIFY: 0x2e[11] vs FuelGain messages")
    print("=" * 80)

    fuel_gains = all_results.get("fuel_gain", [])
    self_msgs = all_results.get("0x2e_self", [])
    print(f"  FuelGain messages: {len(fuel_gains)}")
    print(f"  0x2e self messages: {len(self_msgs)}")

    if fuel_gains and self_msgs:
        match_count = 0
        total_checked = 0
        for fg in fuel_gains[:100]:
            fg_ts = fg["timestamp_ms"]
            fg_fuel = fg["fuel_total"]
            assert isinstance(fg_ts, int) and isinstance(fg_fuel, int)

            closest = None
            closest_delta = 999999
            for s in self_msgs:
                delta = abs(s["timestamp_ms"] - fg_ts)
                assert isinstance(delta, int)
                if delta < closest_delta:
                    closest_delta = delta
                    closest = s

            if closest is not None and closest_delta < 200:
                total_checked += 1
                b11 = closest["byte11"]
                b8 = closest["byte8"]
                assert isinstance(b11, int) and isinstance(b8, int)
                fuel_low = fg_fuel & 0xFF
                fuel_high = (fg_fuel >> 8) & 0xFF
                match_low = b11 == fuel_low
                match_high = b8 == fuel_high
                if match_low:
                    match_count += 1
                if total_checked <= 20:
                    print(
                        f"    FuelGain={fg_fuel} (low=0x{fuel_low:02x} high=0x{fuel_high:02x}) "
                        f"byte11=0x{b11:02x} byte8=0x{b8:02x} "
                        f"low_match={match_low} high_match={match_high} delta={closest_delta}ms"
                    )

        print(f"\n  byte11 == fuel & 0xFF: {match_count}/{total_checked}")
        if total_checked > 0:
            print(f"  Match rate: {100 * match_count / total_checked:.1f}%")

    # ================================================================
    # 4. Combat_data viewport offset verification
    # ================================================================
    print()
    print("=" * 80)
    print("VERIFY: combat_data viewport offset")
    print("=" * 80)

    # Load combat data from wire_byte_analysis.json
    wd = json.loads(Path("wire_byte_analysis.json").read_text())
    combat_list = wd.get("COMBAT_HIT", [])

    # For each combat_hit with a 0x3d match, compute the viewport offset
    offsets: list[tuple[int, int]] = []
    for ch in combat_list:
        cd = ch.get("combat_data_bytes", [])
        aid = ch.get("attacker_id")
        ts = ch.get("timestamp_ms")
        if (
            not isinstance(cd, list)
            or len(cd) < 6
            or not isinstance(aid, int)
            or not isinstance(ts, int)
        ):
            continue

        # cd[0:2] = attacker map position (proven)
        # cd[2:4] = target "viewport" position
        # Find 0x3d for self (target) at same time
        for u in u3d:
            u_tid = u["tank_id"]
            assert isinstance(u_tid, int)
            if u_tid == 1301 and abs(u["timestamp_ms"] - ts) < 200:
                self_x = u["x"]
                self_y = u["y"]
                assert isinstance(self_x, int) and isinstance(self_y, int)
                offset_x = cd[2] - self_x
                offset_y = cd[3] - self_y
                offsets.append((offset_x, offset_y))
                break

    if offsets:
        offset_counts: dict[tuple[int, int], int] = defaultdict(int)
        for o in offsets:
            offset_counts[o] += 1
        print(f"  Total combat_data offset samples: {len(offsets)}")
        print("  Distinct offsets:")
        for off, count in sorted(offset_counts.items(), key=lambda x: -x[1])[:10]:
            print(f"    offset=({off[0]:+d},{off[1]:+d}): {count}")
    else:
        print("  No offset samples found")

    # ================================================================
    # 5. UNKNOWN subtypes beyond 0x2e/0x3d/0x5a
    # ================================================================
    print()
    print("=" * 80)
    print("ALL UNKNOWN SUBTYPES in wire_byte_analysis.json")
    print("=" * 80)

    unknowns = wd.get("UNKNOWN", [])
    by_subtype: dict[int, int] = defaultdict(int)
    by_subtype_lens: dict[int, set[int]] = defaultdict(set)
    for u in unknowns:
        b = u["raw_bytes"]
        if isinstance(b, list) and b:
            st = b[0]
            by_subtype[st] += 1
            by_subtype_lens[st].add(len(b))

    for st in sorted(by_subtype.keys()):
        lens = sorted(by_subtype_lens[st])
        print(f"  subtype=0x{st:02x} ({st}): {by_subtype[st]} messages, lengths={lens}")


if __name__ == "__main__":
    main()
