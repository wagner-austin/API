"""Archive sweep: do bots shoot at other bots' tiles, and when?

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
agg = Counter()
examples = []

paths = sorted(Path("runs").glob("*/*.capture_session.json"))
for path in paths:
    try:
        result = scan_session(path)
    except Exception:
        continue
    if "reason" in result:
        continue
    names = {}
    teams = {}
    pos = {}
    last_player_shot_ms = 0
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
            teams[dm["tank_id"]] = dm["team"]
        elif mt in (0x28, 0x3D):
            pos[dm["tank_id"]] = (dm["x"], dm["y"])
        elif mt == 0x47:
            wp = dm["waypoints"]
            end = wp[-1] if wp else (dm["start_x"], dm["start_y"])
            pos[dm["tank_id"]] = end
        elif mt == 0x53:
            sid = dm["shooter_id"]
            sname = names.get(sid, "")
            is_bot_shooter = bool(BOT_NAME.match(sname))
            if not is_bot_shooter:
                last_player_shot_ms = ts
                continue
            tgt = (dm["target_x"], dm["target_y"])
            # who stands at the target tile?
            hit_ids = [tid for tid, p in pos.items() if p == tgt and tid != sid]
            hit_bots = [t for t in hit_ids if BOT_NAME.match(names.get(t, ""))]
            hit_players = [t for t in hit_ids if names.get(t) and not BOT_NAME.match(names[t])]
            if hit_players:
                agg["bot_shot_at_player_tile"] += 1
            elif hit_bots:
                agg["bot_shot_at_bot_tile"] += 1
                same_team = any(teams.get(t) == teams.get(sid) for t in hit_bots)
                agg["  same_team_target"] += 1 if same_team else 0
                recent = ts - last_player_shot_ms <= 10000
                agg["  within_10s_of_player_shot"] += 1 if recent else 0
                if len(examples) < 8:
                    examples.append(
                        (
                            path.name,
                            round(ts / 1000),
                            sname,
                            [names.get(t) for t in hit_bots],
                            "recent" if recent else "cold",
                        )
                    )
            else:
                agg["bot_shot_at_empty_or_unknown"] += 1
print(dict(agg))
print("examples:")
for e in examples:
    print("  ", e)
