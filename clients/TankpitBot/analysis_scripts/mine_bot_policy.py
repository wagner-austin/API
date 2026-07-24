"""Sweep 1: crack game-bot policy from the capture archive.

Per-bot event streams from every decodable session, aggregated into
policy evidence: weapon mix, reactivity (fire only when fired upon?),
roam geometry, mine usage, refuel (tier-up) events, teleport-off.

Bots are identified by the 0x21 name pattern <team>-<n> (sd() naming).
"""

from __future__ import annotations

import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

from platform_core.json_utils import load_json_str
from tankpit_bot.protocol import decode_message
from tankpit_bot.sniffer.constants import MSG_MIN_LENGTHS
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode
from tankpit_bot.types import decode_capture_session
from tankpit_bot.validate.wire_timeline import _split_frame_bodies

BOT_NAME = re.compile(r"^(red|purple|blue|orange)-\d+$")
TRACKED = {0x21, 0x28, 0x2E, 0x3D, 0x41, 0x47, 0x4C, 0x4F, 0x53, 0x58, 0x5A, 0x67}


def narrow(obj: object) -> dict:
    if not isinstance(obj, dict):
        raise ValueError("not a dict")
    return obj


def scan_session(path: Path, agg: dict) -> None:
    session = decode_capture_session(narrow(load_json_str(path.read_text(encoding="utf-8"))))
    magic = session["magic"]
    if magic is None:
        agg["skipped_no_magic"] += 1
        return
    reset_xor_state()
    build_global_xor_table(magic)

    names: dict[int, str] = {}
    self_id: int | None = None
    last_pos: dict[int, tuple[int, int, int]] = {}  # id -> (ts, x, y)
    last_hit_ts: dict[int, int] = {}
    last_attacker: dict[int, int] = {}  # victim id -> shooter id
    hits_since_jump: dict[int, int] = {}
    last_move_ts: dict[int, int] = {}  # last 0x47 walk echo per tank
    last_jump_ts: dict[int, int] = {}  # last detected teleport per tank
    viewport: list[int | None] = [None, None]  # own viewport origin
    last_killed_ts: dict[int, int] = {}  # 0x41 victim -> ts
    death_watch: dict[int, tuple[int, int, int]] = {}  # tid -> (death_ts, corpse_x, corpse_y)
    tile_state: dict[tuple[int, int], tuple[int, str]] = {}  # (x,y) -> (ts, state)
    gain_ts: list[int] = []  # own 0x67 timestamps

    def note_tile(ts: int, x: int, y: int, value: int) -> None:
        """Track a tile's container layer: -1 equipment, 0 empty, >0 fuel."""
        state = "equipment" if value == -1 else ("empty" if value == 0 else "fuel")
        prev = tile_state.get((x, y))
        if prev is not None and prev[1] != state:
            dt = ts - prev[0]
            if prev[1] == "empty" and state == "equipment":
                agg["equipment_spawn_transitions"] += 1
                agg["equipment_spawn_gap_min"][min(int(dt // 60000), 10)] += 1
            elif prev[1] == "equipment" and state == "empty":
                agg["equipment_consumed_transitions"] += 1
                near_gain = any(abs(g - ts) <= 5000 for g in gain_ts[-6:])
                agg["equipment_consumed_near_own_gain_5s"] += 1 if near_gain else 0
        if prev is not None and prev[1] == "empty":
            agg["equipment_exposure_min"] += (ts - prev[0]) / 60000.0
        if prev is None and state == "equipment":
            agg["equipment_first_reveals"] += 1
        tile_state[(x, y)] = (ts, state)

    def note_any_pos(tid: int, ts: int, x: int, y: int) -> None:
        watch = death_watch.get(tid)
        if watch is not None and ts >= watch[0] + 21000:
            dist = max(abs(x - watch[1]), abs(y - watch[2]))
            agg["respawn_displacement"][min(dist // 8, 12)] += 1
            agg["respawn_pairs"].append(
                {
                    "session": session["session_id"],
                    "bot": names.get(tid, ""),
                    "corpse": [watch[1], watch[2]],
                    "next_seen": [x, y],
                    "dist": dist,
                    "gap_ms": ts - watch[0],
                }
            )
            del death_watch[tid]
    last_tier: dict[int, tuple[int, int]] = {}  # id -> (ts, tier)
    max_rank: dict[int, int] = {}
    session_bot_shots = 0

    def is_bot(tid: int) -> bool:
        return tid in names and BOT_NAME.match(names[tid]) is not None

    def note_rank(tid: int, rank: int) -> None:
        max_rank[tid] = max(max_rank.get(tid, 0), rank)

    def note_tier(tid: int, ts: int, tier: int) -> None:
        prev = last_tier.get(tid)
        if prev is not None and is_bot(tid) and tier > prev[1]:
            agg["bot_tier_up_events"] += 1
            moved_recently = tid in last_move_ts and ts - last_move_ts[tid] <= 5000
            agg["bot_tier_up_after_move_5s"] += 1 if moved_recently else 0
            jumped_recently = tid in last_jump_ts and ts - last_jump_ts[tid] <= 5000
            agg["bot_tier_up_after_teleport_5s"] += 1 if jumped_recently else 0
            agg["tier_up_jumps"][f"{prev[1]}->{tier}"] += 1
            killed_recently = tid in last_killed_ts and ts - last_killed_ts[tid] <= 40000
            agg["bot_tier_up_after_death_40s"] += 1 if killed_recently else 0
            if killed_recently:
                gap_s = (ts - last_killed_ts[tid]) // 1000
                agg["reactivation_gap_s"][gap_s] += 1
            if not moved_recently and not jumped_recently:
                pos = last_pos.get(tid)
                if pos is None or viewport[0] is None:
                    agg["tier_up_unexplained_unknown"] += 1
                else:
                    stale = ts - pos[0] > 10000
                    inside = (
                        viewport[0] <= pos[1] <= viewport[0] + 15
                        and viewport[1] is not None
                        and viewport[1] <= pos[2] <= viewport[1] + 15
                    )
                    if stale:
                        agg["tier_up_unexplained_stale_pos"] += 1
                    elif inside:
                        agg["tier_up_unexplained_in_viewport"] += 1
                        agg["in_viewport_cases"].append(
                            {
                                "session": session["session_id"],
                                "bot": names.get(tid, ""),
                                "ts": ts,
                                "tier": f"{prev[1]}->{tier}",
                                "pos": [pos[1], pos[2]],
                                "pos_age_ms": ts - pos[0],
                                "last_move_age_ms": (
                                    ts - last_move_ts[tid] if tid in last_move_ts else None
                                ),
                            }
                        )
                    else:
                        agg["tier_up_unexplained_off_viewport"] += 1
        last_tier[tid] = (ts, tier)

    def note_pos(tid: int, ts: int, x: int, y: int, via_walk: bool) -> None:
        note_any_pos(tid, ts, x, y)
        prev = last_pos.get(tid)
        if prev is not None and is_bot(tid):
            dist = max(abs(x - prev[1]), abs(y - prev[2]))
            walked = via_walk or (tid in last_move_ts and last_move_ts[tid] >= prev[0])
            if dist > 3 and not walked:
                agg["bot_teleports"] += 1
                agg["bot_teleport_displacement"][min(dist // 8, 12)] += 1
                last_jump_ts[tid] = ts
                rank = max_rank.get(tid, 0)
                agg["hits_before_teleport"].append((rank, hits_since_jump.get(tid, 0)))
                hits_since_jump[tid] = 0
            elif 1 <= dist <= 3 and not walked:
                agg["bot_small_drift_no_walk"] += 1
            elif dist == 0:
                agg["bot_pos_stationary_syncs"] += 1
        last_pos[tid] = (ts, x, y)

    ordered = sorted(session["messages"], key=lambda m: m["timestamp_ms"])
    t0 = ordered[0]["timestamp_ms"] if ordered else 0
    t1 = ordered[-1]["timestamp_ms"] if ordered else 0
    for msg in ordered:
        if msg["direction"] == "sent":
            continue
        ts = msg["timestamp_ms"]
        for body in _split_frame_bodies(msg["payload"]):
            mt = body[0]
            if mt not in TRACKED:
                continue
            data = xor_decode(body)
            if len(data) < MSG_MIN_LENGTHS.get(mt, 3):
                continue
            try:
                m = decode_message(mt, data)
            except Exception:  # noqa: BLE001 - survey sweep, count and move on
                agg["decode_errors"] += 1
                continue
            k = m["msg_type"]
            if k == 0x21:
                names[m["tank_id"]] = m["name"]
                if self_id is None:
                    self_id = m["tank_id"]
            elif k == 0x28:
                note_rank(m["tank_id"], m["rank"])
                note_tier(m["tank_id"], ts, m["damage_state"])
                note_pos(m["tank_id"], ts, m["x"], m["y"], via_walk=False)
            elif k == 0x2E:
                note_rank(m["tank_id"], m["rank"])
                note_tier(m["tank_id"], ts, m["damage_state"])
            elif k == 0x3D:
                note_rank(m["tank_id"], m["rank"])
                note_tier(m["tank_id"], ts, m["damage_state"])
                note_pos(m["tank_id"], ts, m["x"], m["y"], via_walk=False)
            elif k == 0x47:
                tid = m["tank_id"]
                note_rank(tid, m["rank"])
                note_tier(tid, ts, m["damage_state"])
                last_move_ts[tid] = ts
                if is_bot(tid):
                    agg["bot_moves"] += 1
                    agg["bot_path_tiles"][min(m["path_tiles"], 30)] += 1
                end = m["waypoints"][-1] if m["waypoints"] else (m["start_x"], m["start_y"])
                note_pos(tid, ts, end[0], end[1], via_walk=True)
            elif k == 0x53:
                sid = m["shooter_id"]
                # register the hit on whoever stands at the impact tile
                for tid, (_pts, px, py) in list(last_pos.items()):
                    if px == m["target_x"] and py == m["target_y"] and tid != sid:
                        last_hit_ts[tid] = ts
                        last_attacker[tid] = sid
                        hits_since_jump[tid] = hits_since_jump.get(tid, 0) + 1
                if is_bot(sid):
                    session_bot_shots += 1
                    agg["bot_shot_weapons"][m["weapon"]] += 1
                    rng = max(abs(m["target_x"] - m["source_x"]), abs(m["target_y"] - m["source_y"]))
                    agg["bot_shot_range"][min(rng, 20)] += 1
                    atk = last_attacker.get(sid)
                    if atk is not None and atk in last_pos:
                        ax, ay = last_pos[atk][1], last_pos[atk][2]
                        at_attacker = m["target_x"] == ax and m["target_y"] == ay
                        agg["bot_shot_at_attacker"] += 1 if at_attacker else 0
                        agg["bot_shot_with_known_attacker"] += 1
                    since = ts - last_hit_ts[sid] if sid in last_hit_ts else None
                    if since is None:
                        agg["bot_shot_unprovoked_never_hit"] += 1
                    else:
                        agg["bot_reaction_ms"][min(since // 500, 20)] += 1
                        if since <= 3000:
                            agg["bot_shot_within_3s_of_hit"] += 1
                        elif since <= 10000:
                            agg["bot_shot_3_to_10s"] += 1
                        else:
                            agg["bot_shot_over_10s"] += 1
            elif k == 0x41:
                last_killed_ts[m["victim_id"]] = ts
                if is_bot(m["victim_id"]):
                    agg["bot_deaths"] += 1
                    corpse = last_pos.get(m["victim_id"])
                    if corpse is not None and ts - corpse[0] <= 10000:
                        death_watch[m["victim_id"]] = (ts, corpse[1], corpse[2])
                if not m["is_mine_kill"] and is_bot(m["killer_id"]):
                    agg["bot_kills"] += 1
            elif k == 0x4B:
                if is_bot(m["tank_id"]):
                    agg["bot_mine_placements"] += 1
            elif k == 0x5A:
                viewport[0] = m["viewport_left"]
                viewport[1] = m["viewport_top"]
                for ent in m["entities"]:
                    ex = m["viewport_left"] + ent["col"] - 1
                    ey = m["viewport_top"] + ent["row"] - 1
                    if 0 <= ex < 256 and 0 <= ey < 256:
                        note_tile(ts, ex, ey, ent["cache_value"])
            elif k == 0x4C:
                for entry in m["tanks"]:
                    note_any_pos(entry["tank_id"], ts, entry["x"], entry["y"])
            elif k == 0x4F:
                for cont in m["containers"]:
                    note_tile(ts, cont["x"], cont["y"], cont["volume"])
            elif k == 0x67:
                gain_ts.append(ts)

    bots = sorted({names[t] for t in names if BOT_NAME.match(names[t])})
    agg["sessions"] += 1
    agg["sessions_with_bots"] += 1 if bots else 0
    agg["bot_shots_total"] += session_bot_shots
    agg["session_minutes"] += (t1 - t0) / 60000.0
    for tid, name in names.items():
        if BOT_NAME.match(name):
            agg["bot_rank_max"][max_rank.get(tid, 0)] += 1
            agg["distinct_bot_names"].add(name)


def main() -> None:
    runs = Path(sys.argv[1])
    agg: dict = {
        "sessions": 0,
        "sessions_with_bots": 0,
        "skipped_no_magic": 0,
        "decode_errors": 0,
        "session_minutes": 0.0,
        "distinct_bot_names": set(),
        "bot_rank_max": Counter(),
        "bot_shot_weapons": Counter(),
        "bot_shots_total": 0,
        "bot_shot_within_3s_of_hit": 0,
        "bot_shot_3_to_10s": 0,
        "bot_shot_over_10s": 0,
        "bot_shot_unprovoked_never_hit": 0,
        "bot_reaction_ms": Counter(),
        "bot_shot_range": Counter(),
        "bot_shot_at_attacker": 0,
        "bot_shot_with_known_attacker": 0,
        "bot_moves": 0,
        "bot_path_tiles": Counter(),
        "bot_mine_placements": 0,
        "bot_teleports": 0,
        "hits_before_teleport": [],
        "bot_tier_up_events": 0,
        "bot_tier_up_after_move_5s": 0,
        "bot_tier_up_after_teleport_5s": 0,
        "tier_up_unexplained_in_viewport": 0,
        "tier_up_unexplained_off_viewport": 0,
        "tier_up_unexplained_stale_pos": 0,
        "tier_up_unexplained_unknown": 0,
        "in_viewport_cases": [],
        "tier_up_jumps": Counter(),
        "bot_tier_up_after_death_40s": 0,
        "reactivation_gap_s": Counter(),
        "respawn_displacement": Counter(),
        "respawn_pairs": [],
        "bot_teleport_displacement": Counter(),
        "equipment_first_reveals": 0,
        "equipment_spawn_transitions": 0,
        "equipment_spawn_gap_min": Counter(),
        "equipment_consumed_transitions": 0,
        "equipment_consumed_near_own_gain_5s": 0,
        "equipment_exposure_min": 0.0,
        "bot_small_drift_no_walk": 0,
        "bot_pos_stationary_syncs": 0,
        "bot_deaths": 0,
        "bot_kills": 0,
    }
    paths = []
    for sub in ("bot", "sniff"):
        paths.extend(sorted((runs / sub).glob("*.capture_session.json")))
    for p in paths:
        try:
            scan_session(p, agg)
        except Exception as exc:  # noqa: BLE001 - survey; name the bad file
            print(f"SESSION_ERROR {p.name}: {exc}", file=sys.stderr)

    # hits-before-teleport by rank
    by_rank = defaultdict(Counter)
    for rank, hits in agg["hits_before_teleport"]:
        by_rank[rank][hits] += 1
    out = {
        k: (sorted(v) if isinstance(v, set) else dict(v) if isinstance(v, Counter) else v)
        for k, v in agg.items()
        if k != "hits_before_teleport"
    }
    out["hits_before_teleport_by_rank"] = {r: dict(c) for r, c in sorted(by_rank.items())}
    out["distinct_bot_names"] = sorted(agg["distinct_bot_names"])
    Path(sys.argv[2]).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if k != "distinct_bot_names"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
