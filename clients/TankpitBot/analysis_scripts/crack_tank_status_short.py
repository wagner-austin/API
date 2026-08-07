"""Crack the meaning of TankStatusShort byte 8 (currently dropped).

Wiki footnote claims byte 8 = ``rank_points`` (lb_score low byte),
"proven by 13/13 exact timestamp match with 0x3D byte 11" -- but 0x3D
byte 11 is the ``carrying`` flag (0/1), not a low byte of anything.
That cite is internally inconsistent.

This script grinds the 150-session corpus to find what byte 8 actually
matches. For every 9-byte 0x2E container body that survives length-
based dispatch to ``tank_status_short``, it pairs the tank with the
nearest 0x3D MovementResponse for the same tank_id within +- 2s and
tallies which MovementResponse field equals TSS byte 8.

Usage::

    poetry run python -m analysis_scripts.crack_tank_status_short
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol.decoders.tank import decode_0x2e_message

# Migrated 2026-08-06 onto tankpit_bot.analysis.scan (the typed
# capture-scan owner) - the private base64/XOR-table/frame-walk
# pipeline is deleted; results reproduce exactly.


@dataclass(frozen=True)
class TSSample:
    """One TankStatusShort body (9-byte container)."""

    timestamp_ms: int
    tank_id: int
    damage_state: int
    rank: int
    lb_pos: int
    byte_8: int


@dataclass(frozen=True)
class MRSample:
    """One 0x3D MovementResponse, parsed canonically (Mg.h)."""

    timestamp_ms: int
    tank_id: int
    team: int
    x: int
    y: int
    direction: int
    damage_state: int
    rank: int
    lb_score: int
    carrying: int


def _collect(path: Path) -> tuple[list[TSSample], list[MRSample]]:
    """Walk one session and return (TankStatusShort, MovementResponse) lists."""
    result = scan_session(path)
    if "reason" in result:
        return [], []

    tss: list[TSSample] = []
    mrs: list[MRSample] = []

    for frame in result["frames"]:
        if frame["direction"] != "received" or frame["msg_type"] != 0x2E:
            continue
        decoded = frame["body"]
        if len(decoded) < 1:
            continue
        timestamp = frame["timestamp_ms"]
        try:
            routed = decode_0x2e_message(decoded)
        except Exception:
            continue
        mt = routed.get("msg_type")
        if mt == "tank_status_short":
            # Pull byte 8 raw from the original decoded body
            if len(decoded) >= 9:
                tss.append(
                    TSSample(
                        timestamp_ms=timestamp,
                        tank_id=routed["tank_id"],
                        damage_state=routed["damage_state"],
                        rank=routed["rank"],
                        lb_pos=routed["leaderboard_position"],
                        byte_8=decoded[8],
                    )
                )
        elif mt == 0x3D:
            mrs.append(
                MRSample(
                    timestamp_ms=timestamp,
                    tank_id=routed["tank_id"],
                    team=routed["team"],
                    x=routed["x"],
                    y=routed["y"],
                    direction=routed["direction"],
                    damage_state=routed["damage_state"],
                    rank=routed["rank"],
                    lb_score=routed["lb_score"],
                    carrying=routed["carrying"],
                )
            )
    return tss, mrs


def _nearest(tank_id: int, ts: int, pool: list[MRSample], window_ms: int) -> MRSample | None:
    best: MRSample | None = None
    best_dt = window_ms + 1
    for m in pool:
        if m.tank_id != tank_id:
            continue
        dt = abs(m.timestamp_ms - ts)
        if dt < best_dt:
            best_dt = dt
            best = m
    return best


def _candidate_fields(m: MRSample) -> dict[str, int]:
    return {
        "team": m.team,
        "x": m.x,
        "y": m.y,
        "direction": m.direction,
        "damage_state": m.damage_state,
        "rank": m.rank,
        "carrying": m.carrying,
        "lb_score_hi": (m.lb_score >> 16) & 0xFF,
        "lb_score_mid": (m.lb_score >> 8) & 0xFF,
        "lb_score_lo": m.lb_score & 0xFF,
    }


def main(argv: list[str]) -> int:
    paths = (
        [Path(a) for a in argv]
        if argv
        else sorted(
            list(Path("runs/bot").glob("*.capture_session.json"))
            + list(Path("runs/sniff").glob("*.capture_session.json"))
        )
    )

    all_tss: list[TSSample] = []
    all_mrs: list[MRSample] = []
    for path in paths:
        tss, mrs = _collect(path)
        all_tss.extend(tss)
        all_mrs.extend(mrs)

    print(f"Processed {len(paths)} sessions")
    print(f"Collected {len(all_tss)} TankStatusShort bodies, {len(all_mrs)} 0x3D MR bodies")

    if not all_tss:
        print("\nNo TankStatusShort container fallbacks in corpus -- the fix is working.")
        return 0

    # Distribution of byte 8 values across all TSS samples
    byte_counter: Counter[int] = Counter(t.byte_8 for t in all_tss)
    print(f"\nDistinct byte_8 values: {len(byte_counter)}")
    print(f"Range: {min(byte_counter)}..{max(byte_counter)}")
    print("Top 10:")
    for byte, count in byte_counter.most_common(10):
        pct = 100.0 * count / max(len(all_tss), 1)
        print(f"  {byte:3d} (0x{byte:02x})  {count:6d} ({pct:5.1f}%)")

    # Pairs
    pairs: list[tuple[TSSample, MRSample]] = []
    for t in all_tss:
        m = _nearest(t.tank_id, t.timestamp_ms, all_mrs, window_ms=2000)
        if m is not None:
            pairs.append((t, m))
    print(f"\nPaired {len(pairs)} TSS with a 0x3D MR within +-2s")

    hits: Counter[str] = Counter()
    for t, m in pairs:
        for field, value in _candidate_fields(m).items():
            if t.byte_8 == value:
                hits[field] += 1

    print("\nField equality counts (TSS byte_8 == 0x3D field):")
    for field, count in hits.most_common():
        pct = 100.0 * count / max(len(pairs), 1)
        marker = "  <-- LOCKED" if pct >= 95.0 else ""
        print(f"  {field:14s} {count:6d} / {len(pairs)} ({pct:5.1f}%){marker}")

    # Quick sanity: show a few sample pairs
    print("\nSample pairs (first 5):")
    for t, m in pairs[:5]:
        print(
            f"  tid={t.tank_id:5d}  TSS(byte8={t.byte_8:3d} dmg={t.damage_state} "
            f"rank={t.rank} lb_pos={t.lb_pos})  "
            f"MR(x={m.x} y={m.y} dir={m.direction} dmg={m.damage_state} "
            f"rank={m.rank} lb={m.lb_score} carry={m.carrying})"
        )

    # Sanity check: parse first TSS bodies as Og.h (the JS V['.']
    # tunneled handler) and see whether the byte values land in valid
    # ranges for that schema. Og.h layout:
    #   a[0]   = team
    #   a[1:3] = tank_id (LE u16)
    #   a[3]   = damage_state (0-3)
    #   a[4]   = rank (0-8)
    #   a[5:8] = lb_score (24-bit BE)
    #   a[8]   = promo_state (small enum)
    print("\nRaw TSS bodies parsed AS Og.h vs container (sanity bounds):")
    print(f"  {'hex':24s}  Og.h: team tid    dmg rank lb_score  promo   container: tid    rank")
    sane_og = 0
    sane_container = 0
    samples_dumped = 0
    for path in paths:
        result = scan_session(path)
        if "reason" in result:
            continue
        for frame in result["frames"]:
            if frame["direction"] != "received" or frame["msg_type"] != 0x2E:
                continue
            decoded = frame["body"]
            if len(decoded) != 9:
                continue
            try:
                routed = decode_0x2e_message(decoded)
            except Exception:
                continue
            if routed.get("msg_type") != "tank_status_short":
                continue
            # Og.h interpretation
            og_tid = decoded[1] | (decoded[2] << 8)
            og_dmg = decoded[3]
            og_rank = decoded[4]
            og_promo = decoded[8]
            og_ok = og_dmg <= 3 and og_rank <= 8 and og_promo <= 11
            if og_ok:
                sane_og += 1
            # Container TankStatusShort interpretation
            ct_tid = decoded[2] | (decoded[3] << 8)
            ct_rank = decoded[5]
            ct_ok = ct_rank <= 8
            if ct_ok:
                sane_container += 1
            if samples_dumped < 15:
                og_team = decoded[0] & 0x03
                og_lb = (decoded[5] << 16) | (decoded[6] << 8) | decoded[7]
                print(
                    f"  {decoded.hex():24s}  "
                    f"{og_team}    {og_tid:5d}  {og_dmg}    {og_rank}     "
                    f"{og_lb:8d} {og_promo}        {ct_tid:5d}  {ct_rank}"
                )
                samples_dumped += 1
    print(f"\nOg.h sanity (dmg<=3 AND rank<=8 AND promo<=11): {sane_og}/{len(all_tss)}")
    print(f"Container sanity (rank<=8): {sane_container}/{len(all_tss)}")

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
