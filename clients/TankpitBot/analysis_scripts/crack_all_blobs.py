"""Inventory and crack ALL undecoded fields across ALL message types.

Every container and protocol message type that stores opaque bytes
(status_data, extra_data, info_bytes, combat_data, etc.) gets
cross-referenced against proven fields from other message types.
"""

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    d = json.loads(Path("wire_byte_analysis.json").read_text())

    # All message types
    all_types = sorted(d.keys())
    print("=" * 80)
    print("FULL INVENTORY: every message type and undecoded fields")
    print("=" * 80)
    for t in all_types:
        msgs = d[t]
        print(f"  {t}: {len(msgs)} messages")

    # Build lookup indexes from proven sources
    # 0x3d proven: tank_id, x, y, damage_state, rank, direction, lb>>8, rank_points
    unknowns = d.get("UNKNOWN", [])
    u3d = [x for x in unknowns if x["raw_bytes"][0] == 61]

    u3d_by_tank_ts: dict[int, list[tuple[int, dict[str, object]]]] = {}
    for u in u3d:
        b = u["raw_bytes"]
        tid = b[2] | (b[3] << 8)
        u3d_by_tank_ts.setdefault(tid, []).append((u["timestamp_ms"], u))

    def find_0x3d(tank_id: int, ts: int, window: int = 200) -> dict[str, int] | None:
        entries = u3d_by_tank_ts.get(tank_id, [])
        best = None
        best_delta = window + 1
        for entry_ts, entry in entries:
            delta = abs(entry_ts - ts)
            if delta < best_delta:
                best_delta = delta
                b = entry["raw_bytes"]
                best = {
                    "x": b[4], "y": b[5], "direction": b[6],
                    "damage_state": b[7], "rank": b[8],
                    "lb_high": b[10], "rank_points": b[11],
                }
        return best

    # ================================================================
    # 1. TANK_UPDATE_COMPACT status_data (6 bytes beyond [subtype, flags, tank_id])
    # ================================================================
    compact = d.get("TANK_UPDATE_COMPACT", [])
    extended = d.get("TANK_UPDATE_EXTENDED", [])
    full = d.get("TANK_UPDATE_FULL", [])

    for label, updates in [("COMPACT(6)", compact), ("EXTENDED(10)", extended), ("FULL(11)", full)]:
        if not updates:
            continue
        print(f"\n{'=' * 80}")
        print(f"TANK_UPDATE_{label}: cross-ref status_data vs 0x3d proven fields")
        print("=" * 80)

        matches = 0
        for u in updates:
            tid = u.get("tank_id")
            ts = u.get("timestamp_ms")
            sd = u.get("status_data_bytes", [])
            flags = u.get("flags")
            assert isinstance(tid, int) and isinstance(ts, int) and isinstance(sd, list) and isinstance(flags, int)

            ref = find_0x3d(tid, ts)
            if ref is None:
                continue
            matches += 1
            print(f"  tank={tid} ts={ts} flags=0x{flags:02x} ({flags:08b})")
            print(f"    status_data bytes: {sd}")
            print(f"    status_data hex:   {bytes(sd).hex()}")
            print(f"    0x3d ref: x={ref['x']} y={ref['y']} dir={ref['direction']} dmg={ref['damage_state']} rank={ref['rank']}")
            # Check byte-by-byte
            for i, val in enumerate(sd):
                matches_field = []
                if val == ref["x"]:
                    matches_field.append("x")
                if val == ref["y"]:
                    matches_field.append("y")
                if val == ref["direction"]:
                    matches_field.append("direction")
                if val == ref["damage_state"]:
                    matches_field.append("damage_state")
                if val == ref["rank"]:
                    matches_field.append("rank")
                if val == ref["lb_high"]:
                    matches_field.append("lb_high")
                if val == ref["rank_points"]:
                    matches_field.append("rank_points")
                note = f" <-- matches: {', '.join(matches_field)}" if matches_field else ""
                print(f"      sd[{i}] = {val:3d} (0x{val:02x}){note}")
            print()

        if matches == 0:
            print("  (no 0x3d reference data within timestamp window)")

    # ================================================================
    # 2. POSITION_UPDATE extra_data (7 bytes)
    # ================================================================
    pos_updates = d.get("POSITION_UPDATE", [])
    # Note: position_update was 0 in the single-capture run but may exist in aggregate
    if pos_updates:
        print(f"\n{'=' * 80}")
        print("POSITION_UPDATE: cross-ref extra_data (7 bytes) vs 0x3d")
        print("=" * 80)
        for u in pos_updates[:10]:
            tid = u.get("tank_id")
            ts = u.get("timestamp_ms")
            x = u.get("x")
            y = u.get("y")
            ed = u.get("extra_data_bytes", [])
            assert isinstance(tid, int) and isinstance(ts, int) and isinstance(ed, list)
            ref = find_0x3d(tid, ts)
            print(f"  tank={tid} pos=({x},{y}) extra={ed}")
            if ref:
                for i, val in enumerate(ed):
                    matches_field = []
                    if val == ref["direction"]:
                        matches_field.append("direction")
                    if val == ref["damage_state"]:
                        matches_field.append("damage_state")
                    if val == ref["rank"]:
                        matches_field.append("rank")
                    if val == ref["rank_points"]:
                        matches_field.append("rank_points")
                    note = f" <-- {', '.join(matches_field)}" if matches_field else ""
                    print(f"      ed[{i}] = {val:3d} (0x{val:02x}){note}")
            print()

    # ================================================================
    # 3. 0x2e[7] — correlate with self combat state
    # ================================================================
    u2e = [x for x in unknowns if x["raw_bytes"][0] == 46]
    combat_list = d.get("COMBAT_HIT", [])

    print(f"\n{'=' * 80}")
    print("0x2e[7] (boolean flag): correlate with EVERYTHING")
    print("=" * 80)

    # When is byte7==1? Correlate with:
    # - damage_state (byte4)
    # - byte8 (rank_points) even vs odd
    # - byte11 (unknown counter) ranges
    # - byte12 (unknown 0-4)
    # - proximity to combat_hit events

    b7_one = [u for u in u2e if u["raw_bytes"][7] == 1]
    b7_zero = [u for u in u2e if u["raw_bytes"][7] == 0]
    print(f"  byte7=0: {len(b7_zero)} messages")
    print(f"  byte7=1: {len(b7_one)} messages")

    # Check if byte7 correlates with byte11 range
    b11_when_7is0 = [u["raw_bytes"][11] for u in b7_zero]
    b11_when_7is1 = [u["raw_bytes"][11] for u in b7_one]
    if b11_when_7is0:
        print(f"  byte11 when byte7=0: min={min(b11_when_7is0)} max={max(b11_when_7is0)} mean={sum(b11_when_7is0)/len(b11_when_7is0):.1f}")
    if b11_when_7is1:
        print(f"  byte11 when byte7=1: min={min(b11_when_7is1)} max={max(b11_when_7is1)} mean={sum(b11_when_7is1)/len(b11_when_7is1):.1f}")

    # Check if byte7 flips at session boundaries
    print()
    print("  byte7 timeline (showing all transitions and session gaps):")
    prev_b7 = None
    prev_ts = None
    transition_count = 0
    for u in u2e:
        b7 = u["raw_bytes"][7]
        ts = u["timestamp_ms"]
        gap = ""
        if prev_ts is not None and ts - prev_ts > 10000:
            gap = f" [GAP {(ts-prev_ts)/1000:.0f}s]"
        if b7 != prev_b7:
            transition_count += 1
            if transition_count <= 30:
                b4 = u["raw_bytes"][4]
                b8 = u["raw_bytes"][8]
                b11 = u["raw_bytes"][11]
                b12 = u["raw_bytes"][12]
                print(f"    ts={ts} byte7: {prev_b7}->{b7} dmg={b4} rp={b8} b11={b11} b12={b12}{gap}")
        prev_b7 = b7
        prev_ts = ts
    print(f"  Total transitions: {transition_count}")

    # ================================================================
    # 4. 0x2e[11] — deeper analysis
    # ================================================================
    print(f"\n{'=' * 80}")
    print("0x2e[11]: is it the LOW BYTE of fuel?")
    print("=" * 80)

    # If byte[11] is fuel & 0xFF, then fuel = byte[8]*256 + byte[11] would give us
    # a 16-bit fuel value. But byte[8] = rank_points (proven decreasing).
    # So byte[8] is NOT the high byte of fuel.
    #
    # What if bytes [5:7] or [9:11] encode fuel?
    # byte[5]=1 always, byte[6]=0 always, byte[9]=10 always, byte[10]=1 always
    # These constants could be part of a wider value:
    # bytes [9,10,11] = [10, 1, X] as LE = 10 + 256 + X*65536? No, too large.
    # bytes [5,6] = [1,0] as LE = 1? Could be a flag.

    # Check if byte11 wraps around correlating with FuelGain events
    # We don't have FuelGain in the wire_byte_analysis since those come through
    # the protocol path not the container path.

    # Check deltas between consecutive byte11 values
    print("  byte11 delta analysis:")
    deltas_11: dict[int, int] = defaultdict(int)
    for i in range(1, len(u2e)):
        d_val = u2e[i]["raw_bytes"][11] - u2e[i-1]["raw_bytes"][11]
        # Detect wrapping
        if d_val > 128:
            d_val -= 256
        elif d_val < -128:
            d_val += 256
        deltas_11[d_val] += 1

    print("  Top deltas (adjusted for wrapping):")
    for dv in sorted(deltas_11.keys(), key=lambda x: -deltas_11[x])[:20]:
        print(f"    delta={dv:+4d}: {deltas_11[dv]} times")

    # Check: if byte11 is fuel low byte, the delta should be -6 for teleport (min cost)
    # and -10 for radar scan
    fuel_cost_matches = {
        -6: "teleport_min_cost",
        -10: "radar_cost",
        -8: "diagonal_teleport",
    }
    print()
    print("  Fuel cost signature check:")
    for cost, name in fuel_cost_matches.items():
        count = deltas_11.get(cost, 0)
        print(f"    delta={cost} ({name}): {count} occurrences")

    # ================================================================
    # 5. 0x2e[12]: is it related to byte4 transitions?
    # ================================================================
    print(f"\n{'=' * 80}")
    print("0x2e[12] (0-4): when byte12 > byte4, what happens next?")
    print("=" * 80)

    # Track what happens when byte12 != byte4
    diverge_events: list[tuple[int, int, int, int]] = []
    for i, u in enumerate(u2e):
        b = u["raw_bytes"]
        if b[12] > b[4]:
            # Find next message where byte4 changes
            for j in range(i+1, min(i+10, len(u2e))):
                bn = u2e[j]["raw_bytes"]
                if bn[4] != b[4]:
                    diverge_events.append((b[4], b[12], bn[4], bn[12]))
                    break

    print("  Events where byte12 > byte4: tracking next byte4 change")
    pair_counts: dict[tuple[int, int, int, int], int] = defaultdict(int)
    for ev in diverge_events:
        pair_counts[ev] += 1
    for (b4, b12, next_b4, next_b12), count in sorted(pair_counts.items(), key=lambda x: -x[1])[:15]:
        print(f"    dmg={b4} b12={b12} -> next_dmg={next_b4} next_b12={next_b12}: {count}")

    # ================================================================
    # 6. MOVEMENT bytes [5:8] — the "unknown" 3 bytes
    # ================================================================
    movement_list = d.get("MOVEMENT", [])
    if movement_list:
        print(f"\n{'=' * 80}")
        print("MOVEMENT: bytes [5:8] (3 unknown bytes)")
        print("=" * 80)

        for m in movement_list[:10]:
            b = m.get("raw_bytes", [])
            if isinstance(b, list) and len(b) >= 12:
                print(f"  bytes[5:8] = [{b[5]}, {b[6]}, {b[7]}] = 0x{b[5]:02x} 0x{b[6]:02x} 0x{b[7]:02x}")

        # Aggregate
        b5_vals = sorted(set(m["raw_bytes"][5] for m in movement_list if isinstance(m.get("raw_bytes"), list) and len(m["raw_bytes"]) > 5))
        b6_vals = sorted(set(m["raw_bytes"][6] for m in movement_list if isinstance(m.get("raw_bytes"), list) and len(m["raw_bytes"]) > 6))
        b7_vals = sorted(set(m["raw_bytes"][7] for m in movement_list if isinstance(m.get("raw_bytes"), list) and len(m["raw_bytes"]) > 7))
        print(f"\n  byte[5] unique: {b5_vals}")
        print(f"  byte[6] unique: {b6_vals}")
        print(f"  byte[7] unique: {b7_vals}")

    # ================================================================
    # 7. TANK_LEAVE extra_data (2 bytes)
    # ================================================================
    tank_leaves = d.get("TANK_LEAVE", [])
    if tank_leaves:
        print(f"\n{'=' * 80}")
        print(f"TANK_LEAVE: extra_data (2 bytes) — {len(tank_leaves)} messages")
        print("=" * 80)
        ed_vals: dict[str, int] = defaultdict(int)
        for tl in tank_leaves:
            ed = tl.get("extra_data_hex", "")
            assert isinstance(ed, str)
            ed_vals[ed] += 1
        for ed, count in sorted(ed_vals.items(), key=lambda x: -x[1]):
            print(f"  extra_data={ed}: {count}")

    # ================================================================
    # 8. DEACTIVATION_DEATH extra_data (3 bytes)
    # ================================================================
    deaths = d.get("DEACTIVATION_DEATH", [])
    if deaths:
        print(f"\n{'=' * 80}")
        print(f"DEACTIVATION_DEATH: extra_data (3 bytes) — {len(deaths)} messages")
        print("=" * 80)
        for dd in deaths:
            ed = dd.get("extra_data_hex", "")
            ed_bytes = dd.get("extra_data_bytes", [])
            flags = dd.get("flags")
            print(f"  flags={flags} extra={ed} bytes={ed_bytes}")

    # ================================================================
    # 9. TSS flags byte — what do bits 2-7 encode?
    # ================================================================
    tss_list = d.get("TANK_STATUS_SHORT", [])
    if tss_list:
        print(f"\n{'=' * 80}")
        print("TSS FLAGS: bits 2-7 analysis")
        print("=" * 80)

        # Cross-reference TSS flags upper bits with 0x3d data
        for t in tss_list:
            tid = t["tank_id"]
            ts = t["timestamp_ms"]
            flags = t["flags"]
            dmg = t["damage_state"]
            rank = t["rank"]
            assert isinstance(tid, int) and isinstance(ts, int) and isinstance(flags, int)

            upper = flags & 0xFC  # bits 2-7
            if upper != 0:
                ref = find_0x3d(tid, ts)
                ref_info = ""
                if ref:
                    ref_info = f" 0x3d: dmg={ref['damage_state']} rank={ref['rank']} dir={ref['direction']}"
                print(f"  tank={tid} flags=0x{flags:02x} ({flags:08b}) upper=0x{upper:02x} dmg={dmg} rank={rank}{ref_info}")

    # ================================================================
    # 10. WORLD_STATE blob — what's in the large messages?
    # ================================================================
    world_states = d.get("WORLD_STATE", [])
    if world_states:
        print(f"\n{'=' * 80}")
        print(f"WORLD_STATE: {len(world_states)} messages")
        print("=" * 80)
        for ws in world_states[:3]:
            length = ws.get("length")
            raw = ws.get("raw_bytes", [])
            if isinstance(raw, list):
                print(f"  len={length} first_byte=0x{raw[0]:02x} bytes[1:20]={raw[1:20]}")

    # ================================================================
    # 11. CHUNK_DATA — what's in these?
    # ================================================================
    chunks = d.get("CHUNK_DATA", [])
    if chunks:
        print(f"\n{'=' * 80}")
        print(f"CHUNK_DATA: {len(chunks)} messages")
        print("=" * 80)
        for ch in chunks[:3]:
            length = ch.get("length")
            raw = ch.get("raw_bytes", [])
            if isinstance(raw, list):
                print(f"  len={length} first_byte=0x{raw[0]:02x} bytes[1:30]={raw[1:30]}")

    # ================================================================
    # 12. TIP_NOTIFICATION — game tips, can we read the text?
    # ================================================================
    tips = d.get("TIP_NOTIFICATION", [])
    if tips:
        print(f"\n{'=' * 80}")
        print(f"TIP_NOTIFICATION: {len(tips)} messages")
        print("=" * 80)
        for tip in tips[:5]:
            raw = tip.get("raw_bytes", [])
            if isinstance(raw, list):
                # Try ASCII decode
                ascii_chars = "".join(chr(b) if 32 <= b < 127 else "." for b in raw)
                print(f"  len={len(raw)} ascii=[{ascii_chars}]")

    # ================================================================
    # 13. UNKNOWN 0x5a messages — what are these?
    # ================================================================
    u5a = [x for x in unknowns if x["raw_bytes"][0] == 90]
    if u5a:
        print(f"\n{'=' * 80}")
        print(f"UNKNOWN 0x5a: {len(u5a)} messages")
        print("=" * 80)
        for u in u5a[:3]:
            raw = u.get("raw_bytes", [])
            if isinstance(raw, list):
                print(f"  len={len(raw)} bytes[0:40]={raw[:40]}")
                # Check if it contains embedded coordinate pairs or tank data
                # Look for known tank IDs in the data
                for i in range(0, len(raw)-1):
                    tid_check = raw[i] | (raw[i+1] << 8)
                    if 500 <= tid_check <= 535 or tid_check == 1301:
                        print(f"    possible tank_id {tid_check} at offset {i}")


if __name__ == "__main__":
    main()
