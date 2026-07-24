"""Archive sweep: return fire vs gang-up (teammate-aggro) vs assist, with distances."""
import json
import re
from collections import Counter
from pathlib import Path

from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol import decode_message
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode

BOT_NAME = re.compile(r"^(red|purple|blue|orange)-\d+$")
WINDOW_MS = 10000
agg = Counter()
assist_dist = Counter()
gang_dist = Counter()

def split_frames(payload):
    data = decode_base64_safe(payload)
    if not data:
        return
    off = 0
    while off + 2 < len(data):
        ln = data[off] | (data[off + 1] << 8)
        off += 2
        if ln == 0 or off + ln > len(data):
            break
        yield data[off : off + ln]
        off += ln

for path in sorted(Path("runs").glob("*/*.capture_session.json")):
    try:
        s = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        continue
    if not s.get("magic"):
        continue
    reset_xor_state()
    build_global_xor_table(s["magic"])
    msgs = sorted(s["messages"], key=lambda m: m["timestamp_ms"])
    names = {}
    teams = {}
    pos = {}
    last_hit_ms = {}  # tank_id -> last time a shot landed on its tile
    for m in msgs:
        if m["direction"] != "received":
            continue
        ts = m["timestamp_ms"]
        for body in split_frames(m["payload"]):
            try:
                dm = decode_message(body[0], xor_decode(body))
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
                tgt = (dm["target_x"], dm["target_y"])
                # record hit on whoever stands at the target tile
                victims = [tid for tid, p in pos.items() if p == tgt and tid != sid]
                for v in victims:
                    last_hit_ms[v] = ts
                sname = names.get(sid, "")
                if not BOT_NAME.match(sname):
                    continue
                # bot shot: classify
                hit_players = [t for t in victims if names.get(t) and not BOT_NAME.match(names[t])]
                hit_bots = [t for t in victims if BOT_NAME.match(names.get(t, ""))]
                shooter_hit_recently = ts - last_hit_ms.get(sid, -10**12) <= WINDOW_MS
                teammate_hit_recently = any(
                    ts - last_hit_ms.get(t, -10**12) <= WINDOW_MS
                    for t in last_hit_ms
                    if t != sid and teams.get(t) == teams.get(sid)
                    and BOT_NAME.match(names.get(t, ""))
                )
                sp = pos.get(sid)
                if hit_players:
                    if shooter_hit_recently:
                        agg["at_player_return_fire"] += 1
                    elif teammate_hit_recently:
                        agg["at_player_GANG_UP"] += 1
                        if sp:
                            d = max(abs(sp[0] - tgt[0]), abs(sp[1] - tgt[1]))
                            gang_dist[min(d, 20)] += 1
                    else:
                        agg["at_player_unexplained"] += 1
                elif hit_bots:
                    cross_team = any(teams.get(t) != teams.get(sid) for t in hit_bots)
                    if cross_team:
                        agg["at_enemy_bot_ASSIST"] += 1
                        if sp:
                            d = max(abs(sp[0] - tgt[0]), abs(sp[1] - tgt[1]))
                            assist_dist[min(d, 20)] += 1
print(dict(agg))
print("assist shooter->target Chebyshev dist:", dict(sorted(assist_dist.items())))
print("gang-up shooter->player Chebyshev dist:", dict(sorted(gang_dist.items())))
