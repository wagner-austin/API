"""Crack the final three unknown bytes with better methods.

0x2e[7]: check if it's a rank_points underflow flag
0x2e[12]: cross-reference against 0x3d self damage instead of byte4
0x3d[6] / movement[5]: use 1155 movement messages with waypoints directly
"""

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    d = json.loads(Path("wire_byte_analysis.json").read_text())

    unknowns = d.get("UNKNOWN", [])
    u2e = [x for x in unknowns if x["raw_bytes"][0] == 46]
    u3d = [x for x in unknowns if x["raw_bytes"][0] == 61]
    movement_list = d.get("MOVEMENT", [])

    # ================================================================
    # 0x2e[7]: rank_points underflow hypothesis
    # ================================================================
    print("=" * 80)
    print("0x2e[7]: is it a rank_points underflow flag?")
    print("=" * 80)

    # Hypothesis: byte7=1 when rank_points > 0, byte7=0 after rp wraps past 0
    # Track byte7 alongside byte8 (rank_points) across ALL sessions
    print("  Full byte7 vs byte8 (rank_points) timeline at transition points:")
    prev_b7 = None
    prev_b8 = None
    for u in u2e:
        b = u["raw_bytes"]
        b7, b8 = b[7], b[8]
        if b7 != prev_b7 or (prev_b8 is not None and b8 != prev_b8 and abs(b8 - prev_b8) > 200):
            # Transition in b7 OR large jump in b8 (wrap)
            if prev_b7 is not None:
                kind = ""
                if b7 != prev_b7:
                    kind = f" BYTE7_CHANGE {prev_b7}->{b7}"
                if prev_b8 is not None and abs(b8 - prev_b8) > 200:
                    kind += f" RANK_POINTS_WRAP {prev_b8}->{b8}"
                if kind:
                    print(f"    ts={u['timestamp_ms']} byte7={b7} rp={b8}{kind}")
        prev_b7 = b7
        prev_b8 = b8

    # Direct cross-tab: byte7 vs whether rank_points is "high" (>128) or "low" (<128)
    print()
    print("  byte7 vs rank_points quadrant:")
    quad: dict[tuple[int, str], int] = defaultdict(int)
    for u in u2e:
        b = u["raw_bytes"]
        b7 = b[7]
        b8 = b[8]
        rp_half = "rp>=128" if b8 >= 128 else "rp<128"
        quad[(b7, rp_half)] += 1
    for (b7, rp), count in sorted(quad.items()):
        print(f"    byte7={b7} {rp}: {count}")

    # ================================================================
    # 0x2e[12]: cross-reference with 0x3d self damage_state
    # ================================================================
    print()
    print("=" * 80)
    print("0x2e[12]: cross-ref with 0x3d[7] (self damage from 0x3d)")
    print("=" * 80)

    u3d_self = [x for x in u3d if (x["raw_bytes"][2] | (x["raw_bytes"][3] << 8)) == 1301]
    u3d_self.sort(key=lambda x: x["timestamp_ms"])

    # For each 0x2e message, find closest 0x3d self message
    match_count = 0
    total = 0
    for u in u2e[:500]:  # Sample
        ts = u["timestamp_ms"]
        b4 = u["raw_bytes"][4]
        b12 = u["raw_bytes"][12]

        closest_3d = None
        closest_delta = 999999
        for u3 in u3d_self:
            delta = abs(u3["timestamp_ms"] - ts)
            if delta < closest_delta:
                closest_delta = delta
                closest_3d = u3

        if closest_3d is not None and closest_delta < 100:
            d_3d = closest_3d["raw_bytes"][7]
            total += 1
            if b12 == d_3d:
                match_count += 1
            elif b12 != b4:
                print(
                    f"    ts={ts} byte4(self_dmg)={b4} byte12={b12} 0x3d_dmg={d_3d} delta_ms={closest_delta}"
                )

    print(f"\n  byte12 == 0x3d[7]: {match_count}/{total}")
    if total > 0:
        print(f"  Match rate: {100 * match_count / total:.1f}%")

    # Check: is byte12 just byte4 from the PREVIOUS tick?
    print()
    print("  byte12 == previous_byte4 (1-message lag)?")
    lag1 = sum(
        1 for i in range(1, len(u2e)) if u2e[i]["raw_bytes"][12] == u2e[i - 1]["raw_bytes"][4]
    )
    print(f"    {lag1}/{len(u2e) - 1} ({100 * lag1 / (len(u2e) - 1):.1f}%)")

    # Check: is byte12 == byte4 from 2 messages ago?
    lag2 = sum(
        1 for i in range(2, len(u2e)) if u2e[i]["raw_bytes"][12] == u2e[i - 2]["raw_bytes"][4]
    )
    print(f"  byte12 == byte4[t-2]: {lag2}/{len(u2e) - 2} ({100 * lag2 / (len(u2e) - 2):.1f}%)")

    # Check: is byte12 the NEXT byte4 (lookahead)?
    look1 = sum(
        1 for i in range(len(u2e) - 1) if u2e[i]["raw_bytes"][12] == u2e[i + 1]["raw_bytes"][4]
    )
    print(f"  byte12 == byte4[t+1]: {look1}/{len(u2e) - 1} ({100 * look1 / (len(u2e) - 1):.1f}%)")

    # ================================================================
    # DIRECTION: use movement byte[5] vs last waypoint direction
    # With 1155 samples instead of 68 cross-references
    # ================================================================
    print()
    print("=" * 80)
    print("DIRECTION: movement byte[5] vs last waypoint (1155 samples)")
    print("=" * 80)

    dir_to_b5: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    dir_to_b5_first: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for m in movement_list:
        raw = m.get("raw_bytes", [])
        wp = m.get("waypoints", "")
        if not isinstance(raw, list) or len(raw) < 6 or not wp:
            continue
        b5 = raw[5]
        last_dir = wp[-1]
        first_dir = wp[0]
        dir_to_b5[last_dir][b5] += 1
        dir_to_b5_first[first_dir][b5] += 1

    print("  byte[5] vs LAST waypoint direction:")
    for d_char in sorted(dir_to_b5.keys()):
        counts = dict(sorted(dir_to_b5[d_char].items(), key=lambda x: -x[1]))
        total_dir = sum(counts.values())
        top3 = list(counts.items())[:3]
        top3_str = ", ".join(f"{v}({c}/{total_dir}={100 * c / total_dir:.0f}%)" for v, c in top3)
        print(f"    last='{d_char}': top values: {top3_str}")

    print()
    print("  byte[5] vs FIRST waypoint direction:")
    for d_char in sorted(dir_to_b5_first.keys()):
        counts = dict(sorted(dir_to_b5_first[d_char].items(), key=lambda x: -x[1]))
        total_dir = sum(counts.values())
        top3 = list(counts.items())[:3]
        top3_str = ", ".join(f"{v}({c}/{total_dir}={100 * c / total_dir:.0f}%)" for v, c in top3)
        print(f"    first='{d_char}': top values: {top3_str}")

    # Check if byte[5] is the direction BEFORE the movement (heading at start)
    # vs the direction AFTER (heading at end)
    # If it's the heading at start, it should match the PREVIOUS movement's last waypoint
    print()
    print("  byte[5] vs PREVIOUS movement's last waypoint (heading persistence):")

    # Group movements by tank (using player_id since tank_id may be None)
    by_player: dict[int, list[dict[str, object]]] = defaultdict(list)
    for m in movement_list:
        pid = m.get("player_id")
        if isinstance(pid, int):
            by_player[pid].append(m)

    prev_last_to_b5: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for pid, moves in by_player.items():
        moves.sort(key=lambda x: x["timestamp_ms"])
        for i in range(1, len(moves)):
            prev_wp = moves[i - 1].get("waypoints", "")
            curr_raw = moves[i].get("raw_bytes", [])
            if not prev_wp or not isinstance(curr_raw, list) or len(curr_raw) < 6:
                continue
            prev_last = prev_wp[-1]
            curr_b5 = curr_raw[5]
            prev_last_to_b5[prev_last][curr_b5] += 1

    for d_char in sorted(prev_last_to_b5.keys()):
        counts = dict(sorted(prev_last_to_b5[d_char].items(), key=lambda x: -x[1]))
        total_dir = sum(counts.values())
        top3 = list(counts.items())[:3]
        top3_str = ", ".join(f"{v}({c}/{total_dir}={100 * c / total_dir:.0f}%)" for v, c in top3)
        print(f"    prev_last='{d_char}': next byte[5] top values: {top3_str}")

    # ================================================================
    # Also check: 0x3d[6] for a tank that just moved
    # ================================================================
    print()
    print("  0x3d[6] for tanks that just completed a movement:")
    # Find 0x3d messages within 200ms AFTER a movement ends
    for m in movement_list[:200]:
        wp = m.get("waypoints", "")
        if not wp:
            continue
        m_ts = m["timestamp_ms"]
        m_sx = m.get("start_x")
        m_sy = m.get("start_y")
        last_dir = wp[-1]

        # Compute end position from waypoints
        if m_sx is None or m_sy is None:
            continue
        ex, ey = m_sx, m_sy
        for c in wp:
            if c == "n":
                ey -= 1
            elif c == "s":
                ey += 1
            elif c == "e":
                ex += 1
            elif c == "w":
                ex -= 1

        # Find 0x3d at end position within 500ms
        for u in u3d:
            b = u["raw_bytes"]
            u_ts = u["timestamp_ms"]
            if u_ts < m_ts or u_ts > m_ts + 5000:
                continue
            u_x, u_y = b[4], b[5]
            if abs(u_x - ex) <= 1 and abs(u_y - ey) <= 1:
                print(
                    f"    last_wp='{last_dir}' end=({ex},{ey}) 0x3d pos=({u_x},{u_y}) 0x3d[6]={b[6]}"
                )
                break


if __name__ == "__main__":
    main()
