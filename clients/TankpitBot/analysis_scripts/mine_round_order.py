"""Archive sweep: within-round 0x53 resolution order vs ascending tank id.

Migrated 2026-08-06 onto ``tankpit_bot.analysis.scan`` (the typed
capture-scan owner) - the private load/XOR/frame-walk pipeline is
deleted; results reproduce exactly.
"""

from collections import Counter
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol import decode_message

BURST_GAP_MS = 100
agg = Counter()
violations = []

for path in sorted(Path("runs").glob("*/*.capture_session.json")):
    try:
        result = scan_session(path)
    except Exception:
        continue
    if "reason" in result:
        continue
    shots = []
    for frame in sorted(result["frames"], key=lambda f: f["timestamp_ms"]):
        if frame["direction"] != "received":
            continue
        try:
            dm = decode_message(frame["msg_type"], frame["body"])
        except Exception:
            continue
        if dm["msg_type"] == 0x53:
            shots.append((frame["timestamp_ms"], dm["shooter_id"]))
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
