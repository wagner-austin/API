"""Crack every remaining unknown byte in 0x3d and 0x2e messages.

For each byte position, cross-reference against ALL known sources:
- tank_status_short (damage_state, rank, leaderboard_position, flags, extra_byte)
- tank_registry (team, military_rank, is_bot)
- movement (start_x, start_y, flags)
- combat_hit (attacker_id, weapon_byte)
"""

import json
from pathlib import Path


def main() -> None:
    d = json.loads(Path("wire_byte_analysis.json").read_text())

    tss_list = d.get("TANK_STATUS_SHORT", [])
    registry_list = d.get("TANK_REGISTRY", [])
    unknowns = d.get("UNKNOWN", [])

    u3d = [x for x in unknowns if x["raw_bytes"][0] == 61]
    u2e = [x for x in unknowns if x["raw_bytes"][0] == 46]

    # ================================================================
    # 0x3d[8] vs tank_status_short.rank
    # ================================================================
    print("=" * 80)
    print("0x3d[8] vs tank_status_short.rank")
    print("=" * 80)

    # Build tank_id -> rank from tank_status_short
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

    # Cross-reference
    for tid in sorted(set(tss_by_tank.keys()) & set(u3d_by_tank.keys())):
        tss_ranks = sorted(set(t["rank"] for t in tss_by_tank[tid]))
        u3d_b8s = sorted(set(u["raw_bytes"][8] for u in u3d_by_tank[tid]))
        match = tss_ranks == u3d_b8s
        print(f"  Tank {tid}: TSS.rank={tss_ranks} 0x3d[8]={u3d_b8s} match={match}")

    # ================================================================
    # 0x3d[6] - what are the values? correlate with anything?
    # ================================================================
    print()
    print("=" * 80)
    print("0x3d[6] analysis")
    print("=" * 80)

    all_b6_vals: dict[int, int] = {}
    for u in u3d:
        b6 = u["raw_bytes"][6]
        all_b6_vals[b6] = all_b6_vals.get(b6, 0) + 1
    print(f"  All values with counts: {dict(sorted(all_b6_vals.items()))}")

    # Per-tank b6 values
    for tid in sorted(u3d_by_tank.keys()):
        msgs = u3d_by_tank[tid]
        b6s = [m["raw_bytes"][6] for m in msgs]
        if len(set(b6s)) > 1:
            print(f"  Tank {tid}: b6 CHANGES over time: {b6s}")
        else:
            print(f"  Tank {tid}: b6 constant={b6s[0]}")

    # ================================================================
    # 0x3d[9] - what is it?
    # ================================================================
    print()
    print("=" * 80)
    print("0x3d[9] analysis")
    print("=" * 80)

    all_b9_vals: dict[int, int] = {}
    for u in u3d:
        b9 = u["raw_bytes"][9]
        all_b9_vals[b9] = all_b9_vals.get(b9, 0) + 1
    print(f"  All values with counts: {dict(sorted(all_b9_vals.items()))}")

    # ================================================================
    # 0x3d[10:12] - u16 LE - correlate with leaderboard_position?
    # ================================================================
    print()
    print("=" * 80)
    print("0x3d[10:12] vs tank_status_short.leaderboard_position")
    print("=" * 80)

    for tid in sorted(set(tss_by_tank.keys()) & set(u3d_by_tank.keys())):
        tss_lbs = sorted(set(t["leaderboard_position"] for t in tss_by_tank[tid]))
        u3d_u16s = sorted(
            set(u["raw_bytes"][10] | (u["raw_bytes"][11] << 8) for u in u3d_by_tank[tid])
        )
        match = tss_lbs == u3d_u16s
        print(f"  Tank {tid}: TSS.lb={tss_lbs} 0x3d[10:12]={u3d_u16s} match={match}")

    # Also check: is 0x3d[10:12] = rank_points (field s)?
    # registry has military_rank, not rank_points. No source for rank_points.
    print()
    print("  0x3d[10:12] per tank (checking for patterns):")
    for tid in sorted(u3d_by_tank.keys()):
        msgs = u3d_by_tank[tid]
        u16s = [m["raw_bytes"][10] | (m["raw_bytes"][11] << 8) for m in msgs]
        print(f"    Tank {tid}: {u16s}")

    # ================================================================
    # 0x3d[12] - always 0?
    # ================================================================
    print()
    print("=" * 80)
    print("0x3d[12] analysis")
    print("=" * 80)

    all_b12_vals: dict[int, int] = {}
    for u in u3d:
        b12 = u["raw_bytes"][12]
        all_b12_vals[b12] = all_b12_vals.get(b12, 0) + 1
    print(f"  All values with counts: {dict(sorted(all_b12_vals.items()))}")

    # ================================================================
    # 0x2e SELF bytes [5:13] analysis
    # ================================================================
    print()
    print("=" * 80)
    print("0x2e SELF bytes [5] through [12]")
    print("=" * 80)

    for byte_idx in range(5, 13):
        vals: dict[int, int] = {}
        for u in u2e:
            b = u["raw_bytes"]
            if byte_idx < len(b):
                v = b[byte_idx]
                vals[v] = vals.get(v, 0) + 1
        print(f"  byte[{byte_idx}]: values={dict(sorted(vals.items()))}")

    # Check if 0x2e[10:12] as LE u16 correlates with fuel
    print()
    print("  0x2e[10:12] as LE u16 timeline (first 20):")
    for u in u2e[:20]:
        b = u["raw_bytes"]
        u16 = b[10] | (b[11] << 8)
        ts = u["timestamp_ms"]
        b4 = b[4]
        b8 = b[8]
        print(f"    ts={ts} byte4(dmg)={b4} byte8=0x{b8:02x} u16={u16}")

    # ================================================================
    # 0x3d[1] flags vs tank_status_short flags
    # ================================================================
    print()
    print("=" * 80)
    print("0x3d[1] flags vs tank_status_short flags")
    print("=" * 80)

    for tid in sorted(set(tss_by_tank.keys()) & set(u3d_by_tank.keys())):
        tss_flags = sorted(set(t["flags"] for t in tss_by_tank[tid]))
        u3d_flags = sorted(set(u["raw_bytes"][1] for u in u3d_by_tank[tid]))
        print(
            f"  Tank {tid}: TSS.flags={[hex(f) for f in tss_flags]} 0x3d[1]={[hex(f) for f in u3d_flags]}"
        )

    # Check if lower 2 bits = team
    print()
    print("  0x3d[1] lower 2 bits vs known teams:")
    for tid in sorted(u3d_by_tank.keys()):
        msgs = u3d_by_tank[tid]
        flags = msgs[0]["raw_bytes"][1]
        team_bits = flags & 0x03
        # Find team from registry
        reg_team = None
        for r in registry_list:
            if r.get("tank_id") == tid and not r.get("is_container"):
                reg_team = r.get("team")
                break
        team_names = {0: "red", 1: "purple", 2: "blue", 3: "orange"}
        print(
            f"    Tank {tid}: flags=0x{flags:02x} team_bits={team_bits}={team_names.get(team_bits, '?')} registry_team={reg_team}"
        )

    # ================================================================
    # 0x3d[8] vs tank_registry military_rank
    # ================================================================
    print()
    print("=" * 80)
    print("0x3d[8] vs tank_registry.military_rank")
    print("=" * 80)

    reg_by_tank: dict[int, list[dict[str, object]]] = {}
    for r in registry_list:
        if not r.get("is_container"):
            tid = r["tank_id"]
            assert isinstance(tid, int)
            reg_by_tank.setdefault(tid, []).append(r)

    for tid in sorted(set(reg_by_tank.keys()) & set(u3d_by_tank.keys())):
        reg_ranks = sorted(set(r["military_rank"] for r in reg_by_tank[tid]))
        u3d_b8s = sorted(set(u["raw_bytes"][8] for u in u3d_by_tank[tid]))
        match = reg_ranks == u3d_b8s
        print(f"  Tank {tid}: registry.rank={reg_ranks} 0x3d[8]={u3d_b8s} match={match}")


if __name__ == "__main__":
    main()
