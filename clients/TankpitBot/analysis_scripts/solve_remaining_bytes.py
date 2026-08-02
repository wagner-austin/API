"""Recursively solve remaining unknown bytes.

Uses proven fields as anchors to cross-reference unknowns against
every other value in the dataset. For each unproven byte, tests every
hypothesis by computing correlation with every known field.
"""

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    d = json.loads(Path("wire_byte_analysis.json").read_text())

    tss_list = d.get("TANK_STATUS_SHORT", [])
    combat_list = d.get("COMBAT_HIT", [])
    movement_list = d.get("MOVEMENT", [])
    unknowns = d.get("UNKNOWN", [])

    u3d = [x for x in unknowns if x["raw_bytes"][0] == 61]
    u2e = [x for x in unknowns if x["raw_bytes"][0] == 46]

    # ================================================================
    # SOLVE 0x3d[6]: correlate with movement direction
    # ================================================================
    print("=" * 80)
    print("SOLVING 0x3d[6]: correlate with movement waypoint direction")
    print("=" * 80)

    # Movement messages have waypoints like 'nnnneeee' showing last direction
    # For each movement, find 0x3d for same tank within 100ms and compare b6
    # with the LAST waypoint direction

    dir_map = {"n": "north", "s": "south", "e": "east", "w": "west"}

    # Build movement index by timestamp for fast lookup
    direction_correlations: list[tuple[str, int]] = []
    for mov in movement_list:
        wp = mov.get("waypoints", "")
        if not wp:
            continue
        last_dir = wp[-1]
        mov_ts = mov["timestamp_ms"]
        mov_tid = mov.get("tank_id")  # may be None (needs PlayerIdMapper)
        mov_sx = mov.get("start_x")
        mov_sy = mov.get("start_y")

        # Find 0x3d messages within 100ms
        for u in u3d:
            b = u["raw_bytes"]
            u_ts = u["timestamp_ms"]
            if abs(u_ts - mov_ts) > 100:
                continue
            u_tid = b[2] | (b[3] << 8)
            # Match by position proximity (movement start should be near 0x3d position)
            pos_match = (
                mov_sx is not None
                and mov_sy is not None
                and abs(b[4] - mov_sx) <= 3
                and abs(b[5] - mov_sy) <= 3
            )
            if pos_match:
                direction_correlations.append((last_dir, b[6]))

    # Count which b6 values appear for each direction
    dir_to_b6: dict[str, dict[int, int]] = defaultdict(lambda: defaultdict(int))
    for last_dir, b6 in direction_correlations:
        dir_to_b6[last_dir][b6] += 1

    print(f"  Total direction-b6 correlations: {len(direction_correlations)}")
    for d_char in sorted(dir_to_b6.keys()):
        b6_counts = dict(sorted(dir_to_b6[d_char].items(), key=lambda x: -x[1]))
        print(
            f"  Last waypoint '{d_char}' ({dir_map.get(d_char, '?')}): b6 distribution = {b6_counts}"
        )

    # Also check: does b6 change WHEN position changes?
    print()
    print("  b6 changes aligned with position changes (per tank):")
    u3d_by_tank: dict[int, list[dict[str, object]]] = {}
    for u in u3d:
        b = u["raw_bytes"]
        tid = b[2] | (b[3] << 8)
        u3d_by_tank.setdefault(tid, []).append(u)

    b6_with_pos_change = 0
    b6_without_pos_change = 0
    for tid, msgs in u3d_by_tank.items():
        prev_x, prev_y, prev_b6 = None, None, None
        for m in msgs:
            b = m["raw_bytes"]
            x, y, b6 = b[4], b[5], b[6]
            if prev_x is not None and prev_b6 is not None:
                pos_changed = x != prev_x or y != prev_y
                b6_changed = b6 != prev_b6
                if b6_changed and pos_changed:
                    b6_with_pos_change += 1
                elif b6_changed and not pos_changed:
                    b6_without_pos_change += 1
            prev_x, prev_y, prev_b6 = x, y, b6

    print(f"    b6 changes WITH position change: {b6_with_pos_change}")
    print(f"    b6 changes WITHOUT position change: {b6_without_pos_change}")

    # ================================================================
    # SOLVE 0x3d[10:11]: check if it's rank_points (s field)
    # ================================================================
    print()
    print("=" * 80)
    print("SOLVING 0x3d[10:11]: monotonic decrement test (rank_points?)")
    print("=" * 80)

    # rank_points counts DOWN. Check if u16 is monotonically non-increasing per tank
    for tid in sorted(u3d_by_tank.keys())[:10]:
        msgs = u3d_by_tank[tid]
        u16s = [m["raw_bytes"][10] | (m["raw_bytes"][11] << 8) for m in msgs]
        # Check monotonic non-increasing
        increases = sum(1 for i in range(1, len(u16s)) if u16s[i] > u16s[i - 1])
        decreases = sum(1 for i in range(1, len(u16s)) if u16s[i] < u16s[i - 1])
        same = sum(1 for i in range(1, len(u16s)) if u16s[i] == u16s[i - 1])
        print(
            f"  Tank {tid}: {len(u16s)} values, decreases={decreases} increases={increases} same={same}"
        )
        if len(u16s) <= 30:
            print(f"    values: {u16s}")
        else:
            print(f"    first 10: {u16s[:10]}")
            print(f"    last 10:  {u16s[-10:]}")

    # ================================================================
    # SOLVE 0x2e[7]: correlate with combat timing
    # ================================================================
    print()
    print("=" * 80)
    print("SOLVING 0x2e[7]: when does it become 1?")
    print("=" * 80)

    # byte7 is 0 most of the time, 1 sometimes. When?
    transitions: list[tuple[int, int, int]] = []
    prev_b7 = None
    for u in u2e:
        b = u["raw_bytes"]
        b7 = b[7]
        if prev_b7 is not None and b7 != prev_b7:
            transitions.append((u["timestamp_ms"], prev_b7, b7))
        prev_b7 = b7

    print(f"  Total transitions: {len(transitions)}")
    for ts, old, new in transitions[:20]:
        # Find nearby combat_hit
        nearby_combat = [c for c in combat_list if abs(c["timestamp_ms"] - ts) < 500]
        combat_info = ""
        if nearby_combat:
            combat_info = f" combat_nearby={len(nearby_combat)}"
        # Find nearby 0x2e byte4 (damage) at same time
        nearby_2e = [u for u in u2e if abs(u["timestamp_ms"] - ts) < 5]
        dmg_info = ""
        if nearby_2e:
            dmg_info = f" self_dmg={nearby_2e[0]['raw_bytes'][4]}"
        print(f"  ts={ts} byte7: {old}->{new}{dmg_info}{combat_info}")

    # Check: is byte7==1 correlated with a specific damage_state?
    b7_vs_b4: dict[tuple[int, int], int] = defaultdict(int)
    for u in u2e:
        b = u["raw_bytes"]
        b7_vs_b4[(b[7], b[4])] += 1

    print()
    print("  byte7 vs byte4 (damage_state) cross-tab:")
    for (b7, b4), count in sorted(b7_vs_b4.items()):
        print(f"    byte7={b7} damage={b4}: {count}")

    # ================================================================
    # SOLVE 0x2e[8]: is it rank_points high byte?
    # ================================================================
    print()
    print("=" * 80)
    print("SOLVING 0x2e[8]: what is it? (values and transitions)")
    print("=" * 80)

    b8_vals = sorted(set(u["raw_bytes"][8] for u in u2e))
    print(f"  Unique values: {b8_vals} = {[hex(v) for v in b8_vals]}")

    # When does it change?
    prev_b8 = None
    for u in u2e:
        b = u["raw_bytes"]
        b8 = b[8]
        if prev_b8 is not None and b8 != prev_b8:
            print(
                f"  TRANSITION ts={u['timestamp_ms']}: byte8 {prev_b8} (0x{prev_b8:02x}) -> {b8} (0x{b8:02x})"
            )
            # What else changed at this time?
            prev_b8 = b8
        elif prev_b8 is None:
            prev_b8 = b8

    # ================================================================
    # SOLVE 0x2e[11]: is it fuel?
    # ================================================================
    print()
    print("=" * 80)
    print("SOLVING 0x2e[11]: decrement pattern (fuel? rank_points?)")
    print("=" * 80)

    b11_vals = [u["raw_bytes"][11] for u in u2e]
    print(f"  Range: {min(b11_vals)} to {max(b11_vals)}")
    print(f"  First 30 values: {b11_vals[:30]}")

    # Check if monotonically decreasing
    increases = sum(1 for i in range(1, len(b11_vals)) if b11_vals[i] > b11_vals[i - 1])
    decreases = sum(1 for i in range(1, len(b11_vals)) if b11_vals[i] < b11_vals[i - 1])
    same = sum(1 for i in range(1, len(b11_vals)) if b11_vals[i] == b11_vals[i - 1])
    print(f"  Monotonic: decreases={decreases} increases={increases} same={same}")

    # Check if the increases correspond to fuel gains
    print()
    print("  byte11 jumps UP (potential fuel gain):")
    jump_count = 0
    for i in range(1, len(b11_vals)):
        if b11_vals[i] > b11_vals[i - 1]:
            delta = b11_vals[i] - b11_vals[i - 1]
            ts = u2e[i]["timestamp_ms"]
            jump_count += 1
            if jump_count <= 10:
                print(f"    ts={ts} {b11_vals[i - 1]}->{b11_vals[i]} delta=+{delta}")
    if jump_count > 10:
        print(f"    ... and {jump_count - 10} more jumps")

    # ================================================================
    # SOLVE 0x2e[12]: values 0-4, correlate with damage_state?
    # ================================================================
    print()
    print("=" * 80)
    print("SOLVING 0x2e[12]: values 0-4, vs byte4 (damage_state)")
    print("=" * 80)

    # Check if byte12 == byte4 (redundant damage)
    match_count = sum(1 for u in u2e if u["raw_bytes"][12] == u["raw_bytes"][4])
    total = len(u2e)
    print(f"  byte12 == byte4: {match_count}/{total} ({100 * match_count / total:.1f}%)")

    # Check if byte12 tracks byte4 with a lag
    lag_match = 0
    for i in range(1, len(u2e)):
        if u2e[i]["raw_bytes"][12] == u2e[i - 1]["raw_bytes"][4]:
            lag_match += 1
    print(
        f"  byte12[t] == byte4[t-1] (lagged): {lag_match}/{total - 1} ({100 * lag_match / (total - 1):.1f}%)"
    )

    # Cross-tab
    b12_vs_b4: dict[tuple[int, int], int] = defaultdict(int)
    for u in u2e:
        b = u["raw_bytes"]
        b12_vs_b4[(b[12], b[4])] += 1
    print()
    print("  byte12 vs byte4 cross-tab:")
    print(f"    {'':>6} | byte4=0 | byte4=1 | byte4=2 | byte4=3")
    for b12 in range(5):
        row = f"    b12={b12} |"
        for b4 in range(4):
            count = b12_vs_b4.get((b12, b4), 0)
            row += f" {count:>7} |"
        print(row)


if __name__ == "__main__":
    main()
