"""Archive sweep: within-round 0x53 resolution order vs ascending tank id."""
import json
from collections import Counter
from pathlib import Path

from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol import decode_message
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode

BURST_GAP_MS = 100
agg = Counter()
violations = []

for path in sorted(Path("runs").glob("*/*.capture_session.json")):
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not s.get("magic"):
        continue
    reset_xor_state()
    build_global_xor_table(s["magic"])
    shots = []
    for m in sorted(s["messages"], key=lambda x: x["timestamp_ms"]):
        if m["direction"] != "received":
            continue
        data = decode_base64_safe(m["payload"])
        if not data:
            continue
        off = 0
        while off + 2 < len(data):
            ln = data[off] | (data[off + 1] << 8)
            off += 2
            if ln == 0 or off + ln > len(data):
                break
            body = data[off : off + ln]
            off += ln
            try:
                dm = decode_message(body[0], xor_decode(body))
            except Exception:
                continue
            if dm["msg_type"] == 0x53:
                shots.append((m["timestamp_ms"], dm["shooter_id"]))
    burst = []
    for ts, shooter in shots:
        if burst and ts - burst[-1][0] > BURST_GAP_MS:
            if len({sid for _, sid in burst}) >= 2:
                ids = [sid for _, sid in burst]
                agg["multi_shooter_bursts"] += 1
                if ids == sorted(ids):
                    agg["ascending_id_order"] += 1
                elif len(violations) < 5:
                    violations.append((path.name[:30], ids))
            burst = []
        burst.append((ts, shooter))

print(dict(agg))
print("violations sample:", violations)
