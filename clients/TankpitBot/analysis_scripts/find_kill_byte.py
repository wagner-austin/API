"""Find the kill counter / kill confirmation byte.

Check every byte in 0x2e self messages for changes that correlate
with kill moments. Also check if the "constant" bytes actually change.
Infer kill moments from enemy 0x3d message streams stopping.
"""

import json
from collections import defaultdict
from pathlib import Path


def main() -> None:
    d = json.loads(Path("wire_byte_analysis.json").read_text())

    unknowns = d.get("UNKNOWN", [])
    u2e = [x for x in unknowns if x["raw_bytes"][0] == 46]
    u3d = [x for x in unknowns if x["raw_bytes"][0] == 61]
    combat_list = d.get("COMBAT_HIT", [])
    tss_list = d.get("TANK_STATUS_SHORT", [])

    # ================================================================
    # Step 1: Are the "constant" bytes actually constant?
    # ================================================================
    print("=" * 80)
    print("0x2e 'CONSTANT' BYTES — are they really constant?")
    print("=" * 80)

    for byte_idx in [5, 6, 9, 10]:
        vals: dict[int, int] = defaultdict(int)
        for u in u2e:
            v = u["raw_bytes"][byte_idx]
            vals[v] += 1
        print(f"  byte[{byte_idx}]: {dict(sorted(vals.items()))}")

    # ================================================================
    # Step 2: Infer kill moments from enemy 0x3d streams
    # A kill = enemy goes to damage_state 1 (critical) then disappears
    # ================================================================
    print()
    print("=" * 80)
    print("INFERRED KILLS: enemy 0x3d streams that end at damage=1")
    print("=" * 80)

    u3d_by_tank: dict[int, list[dict[str, object]]] = defaultdict(list)
    for u in u3d:
        b = u["raw_bytes"]
        tid = b[2] | (b[3] << 8)
        if tid != 1301:  # Skip self
            u3d_by_tank[tid].append(u)

    # Sort each tank's messages by timestamp
    for tid in u3d_by_tank:
        u3d_by_tank[tid].sort(key=lambda x: x["timestamp_ms"])

    # Find tanks whose stream ends with damage_state going to 1
    # and whose last message is followed by a gap (no more messages)
    kill_timestamps: list[tuple[int, int]] = []  # (timestamp, tank_id)

    for tid, msgs in u3d_by_tank.items():
        for i in range(1, len(msgs)):
            prev_dmg = msgs[i - 1]["raw_bytes"][7]
            curr_dmg = msgs[i]["raw_bytes"][7]

            # Damage went from 2 -> 1 (medium -> critical) or 1 -> ???
            # Check if the stream STOPS after reaching critical damage
            if curr_dmg == 1 and prev_dmg == 2:
                # Is this the last message, or is there a long gap after?
                if (
                    i == len(msgs) - 1
                    or msgs[i + 1]["timestamp_ms"] - msgs[i]["timestamp_ms"] > 10000
                ):
                    kill_timestamps.append((msgs[i]["timestamp_ms"], tid))

            # Also check: damage went from 1 -> 0 (critical -> full = healed/respawned)
            # after a gap. The transition TO damage_state 0 after being at 1 with a gap
            # indicates death + respawn
            if curr_dmg == 0 and prev_dmg == 1:
                gap = msgs[i]["timestamp_ms"] - msgs[i - 1]["timestamp_ms"]
                if gap > 5000:
                    # The kill happened at the LAST dmg=1 message
                    kill_timestamps.append((msgs[i - 1]["timestamp_ms"], tid))

    kill_timestamps.sort()
    print(f"  Inferred kill events: {len(kill_timestamps)}")
    for ts, tid in kill_timestamps[:20]:
        print(f"    ts={ts} tank={tid}")

    # ================================================================
    # Step 3: Check EVERY 0x2e byte around kill timestamps
    # ================================================================
    if kill_timestamps:
        print()
        print("=" * 80)
        print("0x2e SELF BYTES around inferred kill timestamps")
        print("=" * 80)

        for kill_ts, kill_tid in kill_timestamps[:10]:
            print(f"\n  KILL: tank={kill_tid} at ts={kill_ts}")

            # Find 0x2e messages within 2 seconds before and after
            nearby = [u for u in u2e if abs(u["timestamp_ms"] - kill_ts) < 3000]
            nearby.sort(key=lambda x: x["timestamp_ms"])

            if not nearby:
                print("    (no 0x2e messages nearby)")
                continue

            print(
                f"    {'ts':>16} {'b4':>4} {'b5':>4} {'b6':>4} {'b7':>4} {'b8':>4} {'b9':>4} {'b10':>4} {'b11':>4} {'b12':>4} {'delta':>6}"
            )
            prev_bytes = None
            for u in nearby:
                b = u["raw_bytes"]
                ts = u["timestamp_ms"]
                delta = ts - kill_ts
                marker = " <-- KILL" if abs(delta) < 100 else ""

                # Show which bytes CHANGED from previous
                changes = ""
                if prev_bytes is not None:
                    changed_indices = [i for i in range(4, 13) if b[i] != prev_bytes[i]]
                    if changed_indices:
                        changes = f" changed={changed_indices}"

                print(
                    f"    {ts:>16} {b[4]:>4} {b[5]:>4} {b[6]:>4} {b[7]:>4} {b[8]:>4} {b[9]:>4} {b[10]:>4} {b[11]:>4} {b[12]:>4} {delta:>+6}{marker}{changes}"
                )
                prev_bytes = list(b)

    # ================================================================
    # Step 4: Check combat_hit messages around kill timestamps
    # Does the weapon_byte or combat_data differ on a killing blow?
    # ================================================================
    if kill_timestamps:
        print()
        print("=" * 80)
        print("COMBAT_HIT messages around inferred kills")
        print("=" * 80)

        for kill_ts, kill_tid in kill_timestamps[:10]:
            nearby_combat = [c for c in combat_list if abs(c["timestamp_ms"] - kill_ts) < 5000]
            nearby_combat.sort(key=lambda x: x["timestamp_ms"])

            if not nearby_combat:
                continue

            print(f"\n  KILL: tank={kill_tid} at ts={kill_ts}")
            for c in nearby_combat:
                ts = c["timestamp_ms"]
                delta = ts - kill_ts
                aid = c.get("attacker_id")
                wb = c.get("weapon_byte")
                cd = c.get("combat_data_bytes", [])
                marker = " <-- KILL MOMENT" if abs(delta) < 100 else ""
                print(f"    ts={ts} delta={delta:>+6} attacker={aid} weapon={wb} cd={cd}{marker}")

    # ================================================================
    # Step 5: Look for ANY byte that increments in 0x2e
    # ================================================================
    print()
    print("=" * 80)
    print("0x2e: which bytes EVER INCREMENT?")
    print("=" * 80)

    for byte_idx in range(4, 13):
        inc_count = 0
        dec_count = 0
        for i in range(1, len(u2e)):
            prev = u2e[i - 1]["raw_bytes"][byte_idx]
            curr = u2e[i]["raw_bytes"][byte_idx]
            if curr > prev:
                inc_count += 1
            elif curr < prev:
                dec_count += 1
        if inc_count > 0 or dec_count > 0:
            print(f"  byte[{byte_idx}]: increments={inc_count} decrements={dec_count}")

    # ================================================================
    # Step 6: Check 0x3d for SELF around kills
    # Does our leaderboard position (byte[10]) change?
    # ================================================================
    u3d_self = [x for x in u3d if (x["raw_bytes"][2] | (x["raw_bytes"][3] << 8)) == 1301]
    u3d_self.sort(key=lambda x: x["timestamp_ms"])

    if kill_timestamps and u3d_self:
        print()
        print("=" * 80)
        print("0x3d SELF around kills: leaderboard/rank_points changes")
        print("=" * 80)

        for kill_ts, kill_tid in kill_timestamps[:10]:
            nearby_3d = [u for u in u3d_self if abs(u["timestamp_ms"] - kill_ts) < 5000]
            nearby_3d.sort(key=lambda x: x["timestamp_ms"])

            if not nearby_3d:
                continue

            print(f"\n  KILL: tank={kill_tid} at ts={kill_ts}")
            print(
                f"    {'ts':>16} {'x':>4} {'y':>4} {'dir':>4} {'dmg':>4} {'rank':>4} {'lb':>4} {'rp':>4} {'delta':>6}"
            )
            for u in nearby_3d:
                b = u["raw_bytes"]
                ts = u["timestamp_ms"]
                delta = ts - kill_ts
                marker = " <-- KILL" if abs(delta) < 100 else ""
                print(
                    f"    {ts:>16} {b[4]:>4} {b[5]:>4} {b[6]:>4} {b[7]:>4} {b[8]:>4} {b[10]:>4} {b[11]:>4} {delta:>+6}{marker}"
                )

    # ================================================================
    # Step 7: TSS extra_byte (rank_points) — does it jump on kills?
    # ================================================================
    if kill_timestamps and tss_list:
        print()
        print("=" * 80)
        print("TSS rank_points (extra_byte) around kills")
        print("=" * 80)

        for kill_ts, kill_tid in kill_timestamps[:10]:
            # Find TSS for the killed tank around kill time
            nearby_tss = [
                t
                for t in tss_list
                if t["tank_id"] == kill_tid and abs(t["timestamp_ms"] - kill_ts) < 5000
            ]
            if nearby_tss:
                print(f"\n  KILL: tank={kill_tid} at ts={kill_ts}")
                for t in sorted(nearby_tss, key=lambda x: x["timestamp_ms"]):
                    delta = t["timestamp_ms"] - kill_ts
                    print(
                        f"    ts={t['timestamp_ms']} dmg={t['damage_state']} rank={t['rank']} rp={t.get('extra_byte')} delta={delta:>+6}"
                    )


if __name__ == "__main__":
    main()
