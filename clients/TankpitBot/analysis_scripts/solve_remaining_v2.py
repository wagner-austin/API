"""Round 2: decompose compound bytes, verify with cross-references.

Key insight from round 1: I was treating bytes [10:11] as a u16 but
they might be separate fields. And I was guessing at 0x2e fields.
This script decomposes and cross-references each byte individually.
"""

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    d = json.loads(Path("wire_byte_analysis.json").read_text())

    tss_list = d.get("TANK_STATUS_SHORT", [])
    unknowns = d.get("UNKNOWN", [])

    u3d = [x for x in unknowns if x["raw_bytes"][0] == 61]
    u2e = [x for x in unknowns if x["raw_bytes"][0] == 46]

    # Build indexes
    tss_by_tank: dict[int, list[dict[str, object]]] = {}
    for t in tss_list:
        tid = t["tank_id"]
        assert isinstance(tid, int)
        tss_by_tank.setdefault(tid, []).append(t)

    u3d_by_tank: dict[int, list[dict[str, object]]] = {}
    for u in u3d:
        b = u["raw_bytes"]
        tid = b[2] | (b[3] << 8)
        u3d_by_tank.setdefault(tid, []).append(u)

    # ================================================================
    # HYPOTHESIS: 0x3d[10] and [11] are SEPARATE bytes, not a u16
    # Test: does 0x3d[10] match TSS leaderboard_position >> 8?
    # ================================================================
    print("=" * 80)
    print("TEST: 0x3d[10] == TSS.leaderboard_position >> 8 ?")
    print("=" * 80)

    for tid in sorted(set(tss_by_tank.keys()) & set(u3d_by_tank.keys())):
        tss_lb_highs = sorted(set(t["leaderboard_position"] >> 8 for t in tss_by_tank[tid]))
        # Need to get raw byte[10], not the u16
        u3d_b10s = sorted(set(u["raw_bytes"][10] for u in u3d_by_tank[tid]))
        match = tss_lb_highs == u3d_b10s
        tss_lbs = sorted(set(t["leaderboard_position"] for t in tss_by_tank[tid]))
        print(f"  Tank {tid}: TSS.lb={tss_lbs} lb>>8={tss_lb_highs} 0x3d[10]={u3d_b10s} match={match}")

    # ================================================================
    # HYPOTHESIS: 0x3d[11] == TSS.extra_byte (rank_points)?
    # Cross-reference at matching timestamps
    # ================================================================
    print()
    print("=" * 80)
    print("TEST: 0x3d[11] vs TSS.extra_byte at same timestamp")
    print("=" * 80)

    match_count = 0
    total_count = 0
    for tid in sorted(set(tss_by_tank.keys()) & set(u3d_by_tank.keys())):
        for tss_msg in tss_by_tank[tid]:
            tss_ts = tss_msg["timestamp_ms"]
            tss_extra = tss_msg.get("extra_byte")
            if tss_extra is None:
                continue
            # Find closest 0x3d within 100ms
            closest = None
            closest_delta = 999999
            for u in u3d_by_tank[tid]:
                delta = abs(u["timestamp_ms"] - tss_ts)
                if delta < closest_delta:
                    closest_delta = delta
                    closest = u
            if closest is not None and closest_delta <= 100:
                b11 = closest["raw_bytes"][11]
                total_count += 1
                if b11 == tss_extra:
                    match_count += 1
                else:
                    print(f"  MISMATCH tank={tid} ts_delta={closest_delta}ms: 0x3d[11]={b11} TSS.extra={tss_extra} diff={abs(b11 - tss_extra)}")

    print(f"\n  Result: {match_count}/{total_count} exact matches")
    if total_count > 0:
        print(f"  Match rate: {100*match_count/total_count:.1f}%")

    # ================================================================
    # Wider window: 0x3d[11] vs TSS.extra_byte within 2 seconds
    # ================================================================
    print()
    print("  Wider window (2 seconds):")
    match2 = 0
    total2 = 0
    close_count = 0
    for tid in sorted(set(tss_by_tank.keys()) & set(u3d_by_tank.keys())):
        for tss_msg in tss_by_tank[tid]:
            tss_ts = tss_msg["timestamp_ms"]
            tss_extra = tss_msg.get("extra_byte")
            if tss_extra is None:
                continue
            closest = None
            closest_delta = 999999
            for u in u3d_by_tank[tid]:
                delta = abs(u["timestamp_ms"] - tss_ts)
                if delta < closest_delta:
                    closest_delta = delta
                    closest = u
            if closest is not None and closest_delta <= 2000:
                b11 = closest["raw_bytes"][11]
                total2 += 1
                diff = abs(b11 - tss_extra)
                if diff == 0:
                    match2 += 1
                elif diff <= 3:
                    close_count += 1

    print(f"  Result: {match2}/{total2} exact, {close_count} within +/-3")

    # ================================================================
    # 0x3d[11] monotonic decrease test (per tank, per session)
    # ================================================================
    print()
    print("=" * 80)
    print("TEST: 0x3d[11] monotonically decreasing per tank?")
    print("=" * 80)

    for tid in sorted(u3d_by_tank.keys())[:15]:
        msgs = u3d_by_tank[tid]
        b11s = [m["raw_bytes"][11] for m in msgs]
        increases = sum(1 for i in range(1, len(b11s)) if b11s[i] > b11s[i - 1])
        decreases = sum(1 for i in range(1, len(b11s)) if b11s[i] < b11s[i - 1])
        same = sum(1 for i in range(1, len(b11s)) if b11s[i] == b11s[i - 1])
        b8s = sorted(set(m["raw_bytes"][8] for m in msgs))
        print(f"  Tank {tid}: rank={b8s} b11 dec={decreases} inc={increases} same={same}")
        if increases > 0 and len(b11s) <= 30:
            print(f"    values: {b11s}")

    # ================================================================
    # 0x3d[6] direction: tighter correlation with waypoint endpoints
    # ================================================================
    print()
    print("=" * 80)
    print("TEST: 0x3d[6] direction granularity")
    print("=" * 80)

    # Check if it could be a bitfield rather than a scalar
    b6_binary: dict[str, int] = defaultdict(int)
    for u in u3d:
        b6 = u["raw_bytes"][6]
        b6_binary[f"{b6:08b}"] = b6_binary.get(f"{b6:08b}", 0) + 1

    print("  b6 as binary, top 20:")
    for bits, count in sorted(b6_binary.items(), key=lambda x: -x[1])[:20]:
        val = int(bits, 2)
        print(f"    {bits} = {val:3d} (0x{val:02x}): {count}")

    # ================================================================
    # 0x2e SELF: separate each byte, check for fuel correlation
    # ================================================================
    print()
    print("=" * 80)
    print("0x2e SELF: check if bytes [8:12] encode fuel")
    print("=" * 80)

    # The wiki says fuel formula: cost = floor(6 * euclidean_distance)
    # Starting fuel by rank: recruit=1000, private=1100, etc.
    # So fuel is a value 0-1800ish

    # If bytes encode fuel as a multi-byte value, what combination works?
    # byte[8] ranges 0-255 (full range, mostly decreasing)
    # byte[9] = always 10 (0x0a)
    # byte[10] = always 1
    # byte[11] = 0-255 (mostly decreasing with jumps up)

    # Try: fuel = byte[8] * 256 + byte[11]? Or byte[8] + byte[11] * 256?
    # Or fuel = byte[11] + byte[8] * 256?

    print("  Checking byte[8]*256 + byte[11]:")
    vals_8_11 = [u["raw_bytes"][8] * 256 + u["raw_bytes"][11] for u in u2e[:20]]
    print(f"    First 20: {vals_8_11}")

    print("  Checking byte[11] + byte[8]*256:")
    vals_11_8 = [u["raw_bytes"][11] + u["raw_bytes"][8] * 256 for u in u2e[:20]]
    print(f"    First 20: {vals_11_8}")

    # Check decrements
    combo = [u["raw_bytes"][8] * 256 + u["raw_bytes"][11] for u in u2e]
    dec = sum(1 for i in range(1, len(combo)) if combo[i] < combo[i - 1])
    inc = sum(1 for i in range(1, len(combo)) if combo[i] > combo[i - 1])
    same = sum(1 for i in range(1, len(combo)) if combo[i] == combo[i - 1])
    print(f"    byte[8]*256+byte[11] monotonic: dec={dec} inc={inc} same={same}")

    # The 0x2e messages arrive every ~2 seconds. Between them fuel should
    # decrease by cost of movement/teleport/radar. Let's look at deltas.
    print()
    print("  Deltas for byte[8]*256+byte[11] (fuel candidate):")
    deltas: dict[int, int] = defaultdict(int)
    for i in range(1, min(200, len(combo))):
        delta = combo[i] - combo[i - 1]
        deltas[delta] = deltas.get(delta, 0) + 1
    for d_val in sorted(deltas.keys()):
        if deltas[d_val] >= 2:
            print(f"    delta={d_val}: {deltas[d_val]} times")

    # ================================================================
    # 0x2e[12]: deeper cross-reference
    # ================================================================
    print()
    print("=" * 80)
    print("0x2e[12]: check transitions relative to byte[4] (damage)")
    print("=" * 80)

    # Track byte12 and byte4 transitions together
    prev_b4 = None
    prev_b12 = None
    transition_pairs: list[tuple[int, int, int, int]] = []
    for u in u2e:
        b = u["raw_bytes"]
        b4, b12 = b[4], b[12]
        if prev_b4 is not None:
            if b4 != prev_b4 or b12 != prev_b12:
                transition_pairs.append((prev_b4, prev_b12, b4, b12))
        prev_b4, prev_b12 = b4, b12

    print(f"  Total transitions: {len(transition_pairs)}")
    print("  (old_dmg, old_b12) -> (new_dmg, new_b12) counts:")
    pair_counts: dict[tuple[int, int, int, int], int] = defaultdict(int)
    for p in transition_pairs:
        pair_counts[p] += 1
    for (od, ob, nd, nb), count in sorted(pair_counts.items(), key=lambda x: -x[1])[:20]:
        print(f"    ({od},{ob}) -> ({nd},{nb}): {count}")

    # Check: does byte12 LEAD byte4? (byte12 changes first, byte4 follows)
    print()
    print("  Does byte12 predict the NEXT byte4?")
    predict_count = 0
    predict_total = 0
    for i in range(len(u2e) - 1):
        b_now = u2e[i]["raw_bytes"]
        b_next = u2e[i + 1]["raw_bytes"]
        if b_now[4] != b_now[12]:  # disagreement
            predict_total += 1
            if b_next[4] == b_now[12]:
                predict_count += 1
    print(f"    When byte4 != byte12: byte12 predicts next byte4 in {predict_count}/{predict_total}")


if __name__ == "__main__":
    main()
