"""Verify every claim made from reading tpclient.js against actual capture data."""

import json
from collections import defaultdict
from pathlib import Path

from tankpit_bot import _test_hooks
from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode
from tankpit_bot.types import decode_capture_session


def process_all_sessions() -> dict[str, list[dict[str, object]]]:
    """Extract 0x2e self, 0x3d, and FuelGain from all captures."""
    bot_dir = Path("runs/bot")
    paths = sorted(bot_dir.glob("*.capture_session.json"))

    all_data: dict[str, list[dict[str, object]]] = defaultdict(list)
    for path in paths:
        try:
            session_text = _test_hooks.read_text(path)
            from platform_core.json_utils import load_json_str, narrow_json_to_dict

            session_json = narrow_json_to_dict(load_json_str(session_text))
            session = decode_capture_session(session_json)
            magic = session["magic"]
            if magic is None:
                continue
            reset_xor_state()
            build_global_xor_table(magic)

            for msg in session["messages"]:
                if msg["direction"] != "received":
                    continue
                data = decode_base64_safe(msg["payload"])
                if data is None or len(data) < 3:
                    continue
                body = data[2:]
                ts = msg["timestamp_ms"]

                if body[0] == 0x2E:
                    decoded = xor_decode(body)
                    if len(decoded) == 13 and decoded[0] == 0x3D:
                        b = list(decoded)
                        tid = b[2] | (b[3] << 8)
                        all_data["0x3d"].append(
                            {
                                "ts": ts,
                                "tid": tid,
                                "x": b[4],
                                "y": b[5],
                                "dir": b[6],
                                "dmg": b[7],
                                "rank": b[8],
                                "lb_h": b[9],
                                "lb_m": b[10],
                                "lb_l": b[11],
                                "b12": b[12],
                            }
                        )
                    elif len(decoded) == 13 and decoded[0] == 0x2E:
                        b = list(decoded)
                        tid = b[2] | (b[3] << 8)
                        all_data["0x2e"].append(
                            {
                                "ts": ts,
                                "tid": tid,
                                "dmg": b[4],
                                "rank_byte": b[5],
                                "lb_h": b[6],
                                "lb_m": b[7],
                                "lb_l": b[8],
                                "promo": b[9],
                                "has_fuel": b[10],
                                "fuel_lo": b[11],
                                "fuel_hi": b[12],
                                "fuel_value": b[11] + b[12] * 256,
                            }
                        )
                    elif len(decoded) >= 3 and decoded[0] == 0x44:
                        fuel = decoded[1] | (decoded[2] << 8)
                        if len(decoded) >= 4:
                            fuel |= decoded[3] << 16
                        all_data["fuel_gain"].append({"ts": ts, "fuel": fuel})
        except Exception:
            continue

    return dict(all_data)


def main() -> None:
    from platform_core.logging import setup_rich_logging

    setup_rich_logging(level="WARNING")

    d = process_all_sessions()
    u3d = d.get("0x3d", [])
    u2e = d.get("0x2e", [])
    fuel_gains = d.get("fuel_gain", [])

    print(f"Data: {len(u3d)} 0x3d, {len(u2e)} 0x2e, {len(fuel_gains)} FuelGain")

    # ================================================================
    # VERIFY 1: Does direction=32 or 33 appear in 0x3d data?
    # ================================================================
    print()
    print("=" * 80)
    print("VERIFY: direction >= 32 (corpse sprite) in 0x3d data")
    print("=" * 80)

    dir_counts: dict[int, int] = defaultdict(int)
    for u in u3d:
        dir_counts[u["dir"]] += 1

    corpse_count = sum(c for d_val, c in dir_counts.items() if d_val >= 32)
    alive_count = sum(c for d_val, c in dir_counts.items() if d_val < 32)
    print(f"  direction < 32 (alive): {alive_count}")
    print(f"  direction >= 32 (corpse): {corpse_count}")
    print(f"  direction=32: {dir_counts.get(32, 0)}")
    print(f"  direction=33: {dir_counts.get(33, 0)}")
    print(f"  All direction values: {dict(sorted(dir_counts.items()))}")

    # Show context around direction=32/33 transitions
    u3d_by_tank: dict[int, list[dict[str, object]]] = defaultdict(list)
    for u in u3d:
        tid = u["tid"]
        assert isinstance(tid, int)
        u3d_by_tank[tid].append(u)

    print()
    print("  Tanks with direction >= 32:")
    for tid in sorted(u3d_by_tank.keys()):
        msgs = u3d_by_tank[tid]
        corpse_msgs = [m for m in msgs if m["dir"] >= 32]
        if corpse_msgs:
            print(f"    Tank {tid}: {len(corpse_msgs)} corpse messages out of {len(msgs)}")
            # Show transition into corpse
            for i, m in enumerate(msgs):
                if m["dir"] >= 32 and (i == 0 or msgs[i - 1]["dir"] < 32):
                    prev = msgs[i - 1] if i > 0 else None
                    prev_info = f" prev: dir={prev['dir']} dmg={prev['dmg']}" if prev else ""
                    print(
                        f"      TRANSITION: ts={m['ts']} dir={m['dir']} dmg={m['dmg']} rank={m['rank']}{prev_info}"
                    )

    # ================================================================
    # VERIFY 2: 0x2e fuel = byte[11] + byte[12]*256 vs FuelGain
    # ================================================================
    print()
    print("=" * 80)
    print("VERIFY: 0x2e fuel_value (byte[11]+byte[12]*256) vs FuelGain")
    print("=" * 80)

    match_exact = 0
    match_close = 0
    total_checked = 0

    for fg in fuel_gains[:200]:
        fg_ts = fg["ts"]
        fg_fuel = fg["fuel"]
        assert isinstance(fg_ts, int) and isinstance(fg_fuel, int)

        closest = None
        closest_delta = 999999
        for s in u2e:
            delta = abs(s["ts"] - fg_ts)
            assert isinstance(delta, int)
            if delta < closest_delta:
                closest_delta = delta
                closest = s

        if closest is not None and closest_delta < 500:
            total_checked += 1
            fuel_calc = closest["fuel_value"]
            assert isinstance(fuel_calc, int)
            diff = abs(fuel_calc - fg_fuel)
            if diff == 0:
                match_exact += 1
            elif diff <= 10:
                match_close += 1
            if total_checked <= 30:
                mark = "EXACT" if diff == 0 else f"diff={diff}"
                print(
                    f"    FuelGain={fg_fuel} 0x2e fuel={fuel_calc} ({mark}) delta_ms={closest_delta}"
                )

    print(f"\n  Exact match: {match_exact}/{total_checked}")
    print(f"  Within 10: {match_exact + match_close}/{total_checked}")
    if total_checked > 0:
        print(f"  Exact rate: {100 * match_exact / total_checked:.1f}%")

    # ================================================================
    # VERIFY 3: lb_score 24-bit BE in 0x3d matches TSS
    # ================================================================
    print()
    print("=" * 80)
    print("VERIFY: 0x3d lb_score = 256*(256*byte[9]+byte[10])+byte[11]")
    print("=" * 80)

    # Load TSS data from wire_byte_analysis.json for cross-ref
    wd = json.loads(Path("wire_byte_analysis.json").read_text())
    tss_list = wd.get("TANK_STATUS_SHORT", [])

    # TSS has leaderboard_position as LE u16 and extra_byte
    # JS Og parser: lb_score = 256*(256*a[5]+a[6])+a[7] = 24-bit BE
    # For 0x3d: lb_score = 256*(256*byte[9]+byte[10])+byte[11]

    # For TSS: the raw bytes are [subtype, flags, tid_lo, tid_hi, dmg, rank, lb_lo, lb_hi, extra]
    # TSS leaderboard_position is lb_lo + lb_hi*256 (LE u16)
    # TSS extra_byte is at offset 8

    # Check if 0x3d lb_score relates to TSS values
    for tid in sorted(u3d_by_tank.keys())[:5]:
        msgs = u3d_by_tank[tid]
        lb_scores = []
        for m in msgs[:10]:
            lb = 256 * (256 * m["lb_h"] + m["lb_m"]) + m["lb_l"]
            lb_scores.append(lb)
        print(f"  Tank {tid}: lb_scores (first 10) = {lb_scores}")

    # ================================================================
    # VERIFY 4: 0x2e fuel value at session start = 1100 (Private)
    # ================================================================
    print()
    print("=" * 80)
    print("VERIFY: 0x2e fuel at session start")
    print("=" * 80)

    # Group by session gaps (>30s between messages)
    sessions: list[list[dict[str, object]]] = []
    current: list[dict[str, object]] = []
    prev_ts = 0
    for u in u2e:
        ts = u["ts"]
        assert isinstance(ts, int)
        if prev_ts > 0 and ts - prev_ts > 30000:
            if current:
                sessions.append(current)
            current = []
        current.append(u)
        prev_ts = ts
    if current:
        sessions.append(current)

    print(f"  {len(sessions)} sessions detected")
    for i, sess in enumerate(sessions[:15]):
        first = sess[0]
        fuel = first["fuel_value"]
        dmg = first["dmg"]
        rank = first["rank_byte"]
        print(f"    Session {i}: first fuel_value={fuel} dmg={dmg} rank={rank}")

    # ================================================================
    # VERIFY 5: 0x2e byte[12] = fuel_high byte
    # Check: does byte[12] change when fuel crosses 256 boundary?
    # ================================================================
    print()
    print("=" * 80)
    print("VERIFY: 0x2e byte[12] as fuel high byte")
    print("=" * 80)

    # Track byte[12] transitions
    prev_fuel_hi = None
    prev_fuel_lo = None
    transition_count = 0
    for u in u2e[:500]:
        fhi = u["fuel_hi"]
        flo = u["fuel_lo"]
        assert isinstance(fhi, int) and isinstance(flo, int)
        if prev_fuel_hi is not None and fhi != prev_fuel_hi:
            transition_count += 1
            fuel_before = prev_fuel_lo + prev_fuel_hi * 256
            fuel_after = flo + fhi * 256
            if transition_count <= 20:
                print(
                    f"    fuel_hi change: {prev_fuel_hi}->{fhi} fuel: {fuel_before}->{fuel_after} (delta={fuel_after - fuel_before})"
                )
        prev_fuel_hi = fhi
        prev_fuel_lo = flo

    print(f"  Total fuel_hi transitions: {transition_count}")


if __name__ == "__main__":
    main()
