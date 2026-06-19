"""Verify tank 511 damage state across message types."""

import json
from pathlib import Path


def main() -> None:
    d = json.loads(Path("wire_byte_analysis.json").read_text())
    tss = [x for x in d.get("TANK_STATUS_SHORT", []) if x["tank_id"] == 511]
    u3d = [
        x
        for x in d.get("UNKNOWN", [])
        if x["raw_bytes"][0] == 61 and (x["raw_bytes"][2] | (x["raw_bytes"][3] << 8)) == 511
    ]

    print("Tank 511 - tank_status_short (PROVEN):")
    for t in tss:
        print(f"  ts={t['timestamp_ms']} dmg={t['damage_state']} rank={t['rank']}")

    print()
    print("Tank 511 - 0x3d messages:")
    for u in u3d:
        b = u["raw_bytes"]
        print(f"  ts={u['timestamp_ms']} b4=0x{b[4]:02x} b5=0x{b[5]:02x} b6={b[6]} b7={b[7]}")

    print()
    print("INTERLEAVED TIMELINE:")
    events: list[tuple[int, str, int, int]] = []
    for t in tss:
        events.append((t["timestamp_ms"], "TSS", t["damage_state"], t["rank"]))
    for u in u3d:
        b = u["raw_bytes"]
        events.append((u["timestamp_ms"], "0x3d", b[7], b[6]))
    events.sort()
    for ts, src, dmg, extra in events:
        print(f"  ts={ts} src={src:>4} damage={dmg} extra={extra}")

    # Also check tank 1301 (self) - 0x3d b7 vs 0x2e byte4
    print()
    print("Tank 1301 (self) - 0x3d b7 vs 0x2e byte4 interleaved:")
    u3d_self = [
        x
        for x in d.get("UNKNOWN", [])
        if x["raw_bytes"][0] == 61 and (x["raw_bytes"][2] | (x["raw_bytes"][3] << 8)) == 1301
    ]
    u2e_self = [x for x in d.get("UNKNOWN", []) if x["raw_bytes"][0] == 46]

    events2: list[tuple[int, str, int]] = []
    for u in u3d_self:
        b = u["raw_bytes"]
        events2.append((u["timestamp_ms"], "0x3d_b7", b[7]))
    for u in u2e_self:
        b = u["raw_bytes"]
        events2.append((u["timestamp_ms"], "0x2e_b4", b[4]))
    events2.sort()

    prev_3d = None
    prev_2e = None
    for ts, src, val in events2[:40]:
        marker = ""
        if src == "0x3d_b7":
            if prev_3d is not None and val != prev_3d:
                marker = " <-- CHANGE"
            prev_3d = val
        else:
            if prev_2e is not None and val != prev_2e:
                marker = " <-- CHANGE"
            prev_2e = val
        agree = ""
        if prev_3d is not None and prev_2e is not None:
            agree = f" (agree={prev_3d == prev_2e})"
        print(f"  ts={ts} {src:>8}={val}{agree}{marker}")


if __name__ == "__main__":
    main()
