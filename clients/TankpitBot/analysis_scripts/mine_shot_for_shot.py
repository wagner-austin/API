"""Archive test of the shot-for-shot contract: hits taken vs returns fired, and stop->stop gaps.

Migrated 2026-08-06 onto ``tankpit_bot.analysis.scan`` (the typed
capture-scan owner) - the private load/XOR/frame-walk pipeline is
deleted; results reproduce exactly.
"""

import re
from collections import Counter
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol import decode_message

BOT_NAME = re.compile(r"^(red|purple|blue|orange)-\d+$")
ratio_buckets = Counter()
stop_gap_buckets = Counter()
total_hits_on_bots = 0
total_bot_returns = 0

for path in sorted(Path("runs").glob("*/*.capture_session.json")):
    try:
        result = scan_session(path)
    except Exception:
        continue
    if "reason" in result:
        continue
    names = {}
    pos = {}
    hits_taken = Counter()
    shots_fired = Counter()
    last_hit_on = {}
    last_shot_by = {}
    for frame in sorted(result["frames"], key=lambda f: f["timestamp_ms"]):
        if frame["direction"] != "received":
            continue
        ts = frame["timestamp_ms"]
        try:
            dm = decode_message(frame["msg_type"], frame["body"])
        except Exception:
            continue
        mt = dm["msg_type"]
        if mt == 0x21:
            names[dm["tank_id"]] = dm["name"]
        elif mt in (0x28, 0x3D):
            pos[dm["tank_id"]] = (dm["x"], dm["y"])
        elif mt == 0x47:
            wp = dm["waypoints"]
            end = wp[-1] if wp else (dm["start_x"], dm["start_y"])
            pos[dm["tank_id"]] = end
        elif mt == 0x53:
            sid = dm["shooter_id"]
            tgt = (dm["target_x"], dm["target_y"])
            for tid, p in pos.items():
                if p == tgt and tid != sid and BOT_NAME.match(names.get(tid, "")):
                    hits_taken[tid] += 1
                    last_hit_on[tid] = ts
            if BOT_NAME.match(names.get(sid, "")):
                shots_fired[sid] += 1
                last_shot_by[sid] = ts
    for tid, taken in hits_taken.items():
        fired = shots_fired.get(tid, 0)
        total_hits_on_bots += taken
        total_bot_returns += fired
        if taken >= 3:
            ratio = fired / taken
            ratio_buckets[round(ratio * 4) / 4] += 1
        # stop->stop: bot's last shot relative to last hit it took
        if tid in last_shot_by and tid in last_hit_on:
            gap_s = (last_shot_by[tid] - last_hit_on[tid]) / 1000
            stop_gap_buckets[max(min(round(gap_s), 10), -10)] += 1

print(
    f"total hits on bots: {total_hits_on_bots}, total bot shots: {total_bot_returns}, "
    f"global ratio {total_bot_returns / max(total_hits_on_bots, 1):.2f}"
)
print("per-bot fired/taken ratio (engagements with >=3 hits):", dict(sorted(ratio_buckets.items())))
print(
    "last-bot-shot minus last-hit-taken, seconds (clamped +/-10):",
    dict(sorted(stop_gap_buckets.items())),
)
