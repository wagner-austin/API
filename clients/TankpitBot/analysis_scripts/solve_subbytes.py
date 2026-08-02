"""Check for sub-byte fields (bitfields) and coordinate pairs.

Game protocols commonly pack multiple fields into single bytes using
bit masking. Also check if combat_data contains coordinate pairs.
"""

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    d = json.loads(Path("wire_byte_analysis.json").read_text())

    tss_list = d.get("TANK_STATUS_SHORT", [])
    combat_list = d.get("COMBAT_HIT", [])
    unknowns = d.get("UNKNOWN", [])

    u3d = [x for x in unknowns if x["raw_bytes"][0] == 61]
    u2e = [x for x in unknowns if x["raw_bytes"][0] == 46]

    u3d_by_tank: dict[int, list[dict[str, object]]] = {}
    for u in u3d:
        b = u["raw_bytes"]
        tid = b[2] | (b[3] << 8)
        u3d_by_tank.setdefault(tid, []).append(u)

    # ================================================================
    # COMBAT_DATA: are bytes [0:6] coordinate pairs?
    # ================================================================
    print("=" * 80)
    print("COMBAT_DATA: checking if bytes are coordinate pairs")
    print("=" * 80)

    print("  Each combat_hit, with 0x3d positions at same time:")
    for ch in combat_list:
        ts = ch["timestamp_ms"]
        aid = ch.get("attacker_id")
        assert isinstance(aid, int)
        cd = ch.get("combat_data_bytes", [])
        assert isinstance(cd, list)
        wb = ch.get("weapon_byte")

        # Find 0x3d for attacker within 100ms
        attacker_pos = None
        for u in u3d:
            b = u["raw_bytes"]
            u_tid = b[2] | (b[3] << 8)
            if u_tid == aid and abs(u["timestamp_ms"] - ts) < 200:
                attacker_pos = (b[4], b[5])
                break

        # Find 0x3d for self (1301) within 100ms
        self_pos = None
        for u in u3d:
            b = u["raw_bytes"]
            u_tid = b[2] | (b[3] << 8)
            if u_tid == 1301 and abs(u["timestamp_ms"] - ts) < 200:
                self_pos = (b[4], b[5])
                break

        cd_pairs = []
        for i in range(0, len(cd) - 1, 2):
            cd_pairs.append((cd[i], cd[i + 1]))

        print(f"\n  ts={ts} attacker={aid} weapon={wb}")
        print(f"    combat_data pairs: {cd_pairs}")
        if attacker_pos:
            print(f"    attacker 0x3d pos: {attacker_pos}")
            if len(cd_pairs) >= 1:
                p = cd_pairs[0]
                dx = abs(p[0] - attacker_pos[0])
                dy = abs(p[1] - attacker_pos[1])
                print(f"    cd[0:2] vs attacker: delta=({dx},{dy}) match={dx <= 1 and dy <= 1}")
        if self_pos:
            print(f"    self 0x3d pos:     {self_pos}")
            if len(cd_pairs) >= 2:
                p = cd_pairs[1]
                dx = abs(p[0] - self_pos[0])
                dy = abs(p[1] - self_pos[1])
                print(f"    cd[2:4] vs self:    delta=({dx},{dy}) match={dx <= 1 and dy <= 1}")
            if len(cd_pairs) >= 3:
                p = cd_pairs[2]
                dx = abs(p[0] - self_pos[0])
                dy = abs(p[1] - self_pos[1])
                print(f"    cd[4:6] vs self:    delta=({dx},{dy}) match={dx <= 1 and dy <= 1}")

    # ================================================================
    # 0x3d[1] FLAGS: bit-by-bit analysis beyond team
    # ================================================================
    print()
    print("=" * 80)
    print("0x3d[1] FLAGS: per-bit analysis")
    print("=" * 80)

    bit_vals: dict[int, dict[int, int]] = {i: defaultdict(int) for i in range(8)}
    for u in u3d:
        flags = u["raw_bytes"][1]
        for bit in range(8):
            val = (flags >> bit) & 1
            bit_vals[bit][val] += 1

    for bit in range(8):
        counts = dict(bit_vals[bit])
        print(f"  bit {bit} (value {1 << bit}): 0={counts.get(0, 0)} 1={counts.get(1, 0)}")

    # Check bits 2-7 against known fields
    print()
    print("  Unique flag values: ", sorted(set(u["raw_bytes"][1] for u in u3d)))
    print("  (bits 0-1 = team, proven. What about the rest?)")

    # ================================================================
    # 0x3d[6] DIRECTION: per-bit analysis
    # ================================================================
    print()
    print("=" * 80)
    print("0x3d[6] DIRECTION: per-bit analysis")
    print("=" * 80)

    bit_vals_6: dict[int, dict[int, int]] = {i: defaultdict(int) for i in range(8)}
    for u in u3d:
        b6 = u["raw_bytes"][6]
        for bit in range(8):
            val = (b6 >> bit) & 1
            bit_vals_6[bit][val] += 1

    for bit in range(8):
        counts = dict(bit_vals_6[bit])
        print(f"  bit {bit} (value {1 << bit}): 0={counts.get(0, 0)} 1={counts.get(1, 0)}")

    # Check if bits 2-3 map to cardinal direction
    print()
    print("  Bits [4:2] (3-bit field, values 0-7):")
    field_vals: dict[int, int] = defaultdict(int)
    for u in u3d:
        b6 = u["raw_bytes"][6]
        field = (b6 >> 2) & 0x07
        field_vals[field] += 1
    for v in sorted(field_vals.keys()):
        print(f"    {v} ({v:03b}): {field_vals[v]}")

    print()
    print("  Bits [1:0] (2-bit field):")
    field_vals2: dict[int, int] = defaultdict(int)
    for u in u3d:
        b6 = u["raw_bytes"][6]
        field = b6 & 0x03
        field_vals2[field] += 1
    for v in sorted(field_vals2.keys()):
        print(f"    {v} ({v:02b}): {field_vals2[v]}")

    print()
    print("  Bits [5:3] (3-bit field):")
    field_vals3: dict[int, int] = defaultdict(int)
    for u in u3d:
        b6 = u["raw_bytes"][6]
        field = (b6 >> 3) & 0x07
        field_vals3[field] += 1
    for v in sorted(field_vals3.keys()):
        print(f"    {v} ({v:03b}): {field_vals3[v]}")

    # ================================================================
    # 0x2e[12]: per-bit analysis
    # ================================================================
    print()
    print("=" * 80)
    print("0x2e[12]: per-bit analysis (values 0-4)")
    print("=" * 80)

    bit_vals_12: dict[int, dict[int, int]] = {i: defaultdict(int) for i in range(8)}
    for u in u2e:
        b12 = u["raw_bytes"][12]
        for bit in range(8):
            val = (b12 >> bit) & 1
            bit_vals_12[bit][val] += 1

    for bit in range(8):
        counts = dict(bit_vals_12[bit])
        if counts.get(1, 0) > 0:
            print(f"  bit {bit} (value {1 << bit}): 0={counts.get(0, 0)} 1={counts.get(1, 0)}")

    # ================================================================
    # 0x2e[7]: per-bit vs damage state
    # ================================================================
    print()
    print("=" * 80)
    print("0x2e[7]: per-bit analysis")
    print("=" * 80)

    bit_vals_7: dict[int, dict[int, int]] = {i: defaultdict(int) for i in range(8)}
    for u in u2e:
        b7 = u["raw_bytes"][7]
        for bit in range(8):
            val = (b7 >> bit) & 1
            bit_vals_7[bit][val] += 1

    for bit in range(8):
        counts = dict(bit_vals_7[bit])
        if counts.get(1, 0) > 0:
            print(f"  bit {bit} (value {1 << bit}): 0={counts.get(0, 0)} 1={counts.get(1, 0)}")

    # ================================================================
    # TSS flags: per-bit analysis
    # ================================================================
    print()
    print("=" * 80)
    print("TSS flags: per-bit analysis")
    print("=" * 80)

    tss_bit_vals: dict[int, dict[int, int]] = {i: defaultdict(int) for i in range(8)}
    for t in tss_list:
        flags = t["flags"]
        assert isinstance(flags, int)
        for bit in range(8):
            val = (flags >> bit) & 1
            tss_bit_vals[bit][val] += 1

    for bit in range(8):
        counts = dict(tss_bit_vals[bit])
        print(f"  bit {bit} (value {1 << bit}): 0={counts.get(0, 0)} 1={counts.get(1, 0)}")

    print()
    print("  TSS flags unique: ", sorted(set(t["flags"] for t in tss_list)))
    # Check: are TSS flags bits 0-1 also team?
    print("  TSS flags & 0x03 vs known team:")
    for t in tss_list[:5]:
        tid = t["tank_id"]
        flags = t["flags"]
        assert isinstance(flags, int)
        team_bits = flags & 0x03
        print(f"    tank={tid} flags=0x{flags:02x} ({flags:08b}) team_bits={team_bits}")

    # ================================================================
    # Check combat_data[0:2] for sub-byte structure
    # Are high bits flags and low bits coordinates?
    # ================================================================
    print()
    print("=" * 80)
    print("COMBAT_DATA bytes: per-bit analysis for byte[0]")
    print("=" * 80)

    cd_bit_vals: dict[int, dict[int, int]] = {i: defaultdict(int) for i in range(8)}
    for ch in combat_list:
        cd = ch.get("combat_data_bytes", [])
        assert isinstance(cd, list)
        if len(cd) > 0:
            b0 = cd[0]
            for bit in range(8):
                val = (b0 >> bit) & 1
                cd_bit_vals[bit][val] += 1

    for bit in range(8):
        counts = dict(cd_bit_vals[bit])
        print(f"  cd[0] bit {bit}: 0={counts.get(0, 0)} 1={counts.get(1, 0)}")


if __name__ == "__main__":
    main()
