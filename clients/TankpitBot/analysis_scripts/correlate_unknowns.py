"""Correlate unknown 13-byte messages against proven fields.

No guessing. Every field claim is backed by cross-reference with
a known, proven message type at the same timestamp for the same tank.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    data = json.loads(Path("wire_byte_analysis.json").read_text())

    tss_list = data.get("TANK_STATUS_SHORT", [])
    pos_list = data.get("POSITION_UPDATE", [])
    combat_list = data.get("COMBAT_HIT", [])
    movement_list = data.get("MOVEMENT", [])
    unknowns = data.get("UNKNOWN", [])

    u2e = [x for x in unknowns if x["raw_bytes"][0] == 46]
    u3d = [x for x in unknowns if x["raw_bytes"][0] == 61]
    u5a = [x for x in unknowns if x["raw_bytes"][0] == 90]

    print("Data sources:")
    print(f"  tank_status_short: {len(tss_list)}")
    print(f"  position_update (0x24): {len(pos_list)}")
    print(f"  combat_hit: {len(combat_list)}")
    print(f"  movement: {len(movement_list)}")
    print(f"  unknown 0x2e: {len(u2e)}")
    print(f"  unknown 0x3d: {len(u3d)}")
    print(f"  unknown 0x5a: {len(u5a)}")
    print()

    # ================================================================
    # CORRELATION 1: 0x3d byte7 vs tank_status_short.damage_state
    # ================================================================
    print("=" * 80)
    print("CORRELATION 1: 0x3d[7] vs tank_status_short.damage_state")
    print("=" * 80)
    matches_found = 0
    matches_agree = 0

    for ts_msg in tss_list:
        tid = ts_msg["tank_id"]
        ts = ts_msg["timestamp_ms"]
        dmg = ts_msg["damage_state"]
        rank = ts_msg["rank"]

        for u in u3d:
            b = u["raw_bytes"]
            u_tid = b[2] | (b[3] << 8)
            u_ts = u["timestamp_ms"]
            if u_tid == tid and abs(u_ts - ts) <= 2:
                matches_found += 1
                agree = b[7] == dmg
                if agree:
                    matches_agree += 1
                mark = "OK" if agree else "MISMATCH"
                print(f"  [{mark}] tid={tid} ts_delta={u_ts-ts}ms")
                print(f"    tank_status_short: dmg={dmg} rank={rank}")
                print(f"    0x3d:              b7={b[7]} b4=0x{b[4]:02x} b5=0x{b[5]:02x} b6=0x{b[6]:02x}")

    print(f"\nResult: {matches_agree}/{matches_found} matches agree on damage_state")
    print()

    # ================================================================
    # CORRELATION 2: 0x3d byte4/byte5 vs movement start position
    # ================================================================
    print("=" * 80)
    print("CORRELATION 2: 0x3d[4,5] vs movement.start_x/start_y")
    print("=" * 80)
    pos_matches = 0
    pos_agree = 0

    for mov in movement_list:
        mov_tid_raw = mov.get("player_id")
        sx = mov.get("start_x")
        sy = mov.get("start_y")
        mov_ts = mov["timestamp_ms"]

        for u in u3d:
            b = u["raw_bytes"]
            u_tid = b[2] | (b[3] << 8)
            u_ts = u["timestamp_ms"]
            if abs(u_ts - mov_ts) <= 100:
                pos_matches += 1
                print(f"  movement: ts={mov_ts} player_id={mov_tid_raw} pos=({sx},{sy}) waypoints={mov.get('waypoints','')[:10]}")
                print(f"  0x3d:     ts={u_ts} tid={u_tid} b4=0x{b[4]:02x}={b[4]} b5=0x{b[5]:02x}={b[5]}")
                if b[4] == sx or b[5] == sy:
                    pos_agree += 1
                    print("    ^^ POSITION MATCH")
                print()

    print(f"Result: {pos_agree}/{pos_matches} position correlations")
    print()

    # ================================================================
    # CORRELATION 3: 0x2e self-messages - byte4 vs combat damage
    # ================================================================
    print("=" * 80)
    print("CORRELATION 3: 0x2e self-messages byte4 timeline")
    print("=" * 80)
    print("All 0x2e messages are for our tank (1301). Tracking byte4 changes:")
    print()
    prev_b4 = None
    for u in u2e:
        b = u["raw_bytes"]
        b4 = b[4]
        u16 = b[10] | (b[11] << 8)
        b8 = b[8]
        ts = u["timestamp_ms"]
        if b4 != prev_b4:
            print(f"  CHANGE ts={ts} byte4={prev_b4}->{b4} byte8=0x{b8:02x} u16={u16}")
            prev_b4 = b4
        # Check if combat_hit happened near this timestamp
        for ch in combat_list:
            if abs(ch["timestamp_ms"] - ts) <= 5:
                print(f"    ^^ combat_hit at ts={ch['timestamp_ms']} weapon={ch.get('weapon_byte')} attacker={ch.get('attacker_id')}")

    # ================================================================
    # CORRELATION 4: 0x3d byte structure - all fields dump per tank
    # ================================================================
    print()
    print("=" * 80)
    print("CORRELATION 4: 0x3d full byte dump grouped by tank_id")
    print("=" * 80)

    by_tank: dict[int, list[dict[str, object]]] = {}
    for u in u3d:
        b = u["raw_bytes"]
        tid = b[2] | (b[3] << 8)
        by_tank.setdefault(tid, []).append(u)

    for tid in sorted(by_tank.keys()):
        msgs = by_tank[tid]
        print(f"\n  Tank {tid} ({len(msgs)} messages):")
        print(f"    {'ts':>16} {'b1':>4} {'b4':>6} {'b5':>6} {'b6':>4} {'b7':>4} {'b8':>4} {'b9':>4} {'u16':>6} {'b12':>4}")
        for u in msgs:
            b = u["raw_bytes"]
            u16 = b[10] | (b[11] << 8)
            print(f"    {u['timestamp_ms']:>16} 0x{b[1]:02x} 0x{b[4]:02x}={b[4]:>3} 0x{b[5]:02x}={b[5]:>3} {b[6]:>4} {b[7]:>4} {b[8]:>4} {b[9]:>4} {u16:>6} {b[12]:>4}")

    # ================================================================
    # CORRELATION 5: 0x2e self-messages - byte8 vs known rank_points
    # ================================================================
    print()
    print("=" * 80)
    print("CORRELATION 5: 0x2e self-messages byte8 timeline")
    print("=" * 80)
    b8_vals = sorted(set(u["raw_bytes"][8] for u in u2e))
    print(f"  byte8 unique values: {b8_vals} = {[hex(x) for x in b8_vals]}")
    u16_vals = sorted(set(u["raw_bytes"][10]|(u["raw_bytes"][11]<<8) for u in u2e))
    print(f"  u16[10:12] range: {min(u16_vals)} to {max(u16_vals)}")
    print(f"  u16 values (first 20): {u16_vals[:20]}")

    # ================================================================
    # CORRELATION 6: 0x5a messages (world state blobs)
    # ================================================================
    print()
    print("=" * 80)
    print("CORRELATION 6: 0x5a messages")
    print("=" * 80)
    for u in u5a:
        b = u["raw_bytes"]
        print(f"  len={len(b)} first_20_hex={bytes(b[:20]).hex()}")


if __name__ == "__main__":
    main()
