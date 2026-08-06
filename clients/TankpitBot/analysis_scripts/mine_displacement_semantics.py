"""Byte-mine what a teleport displacement RECEIPT implies about the tile.

The proposed displacement-receipt law ("a displaced landing writes an
obstruction observation at the requested tile") needs three questions
answered from the archive before a line of it enters the bot:

1. CAUSE — of displaced requests, how many had mine evidence at the
   requested tile (known before the hop, or revealed later in the same
   session), how many had a tracked tank standing there, how many have
   no visible cause at all (the phantom-obstruction risk)?
2. FALSE NEGATIVES — of EXACT landings, how many happened on a tile
   with a live known mine belief at that moment? The displacement law
   says this must be zero; any hit falsifies the whole premise.
3. DETERMINISM — for request tiles displaced more than once, how many
   distinct landings did the server choose?

Sent teleports come from client-command decode; landings pair to the
self tank's next 0x3D within a window (self id elected by votes:
the tank whose 0x3D most often lands near our aims). Mine truth per
tile is the full statement timeline from 0x4F radar mines/clears,
0x4B placements, and 0x45 detonations.

Usage: python analysis_scripts/mine_displacement_semantics.py <capture ...>
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol import decode_message, try_decode_plaintext_ack
from tankpit_bot.sim.commands import decode_client_command
from tankpit_bot.sniffer.decoders import _is_text_route
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode

LANDING_WINDOW_MS = 3_000
LANDING_MAX_CHEBYSHEV = 8
BODY_FRESH_MS = 5_000


def _iter_frames(data: bytes) -> list[bytes]:
    frames: list[bytes] = []
    offset = 0
    while offset + 2 < len(data):
        length = data[offset] | (data[offset + 1] << 8)
        offset += 2
        if length == 0 or offset + length > len(data):
            return frames
        frames.append(data[offset : offset + length])
        offset += length
    return frames


def _decode_all(session: dict) -> list[tuple[int, str, dict]]:
    """Decode every frame as (t, "cmd"|"msg", decoded)."""
    reset_xor_state()
    build_global_xor_table(session["magic"])
    out: list[tuple[int, str, dict]] = []
    for message in sorted(session["messages"], key=lambda m: m["timestamp_ms"]):
        t = message["timestamp_ms"]
        data = decode_base64_safe(message.get("payload", ""))
        if not data:
            continue
        if message.get("direction") == "sent":
            for body in _iter_frames(data):
                if not body:
                    continue
                try:
                    cmd = dict(decode_client_command(xor_decode(body)))
                except Exception:
                    continue
                out.append((t, "cmd", cmd))
            continue
        for body in _iter_frames(data):
            if not body or try_decode_plaintext_ack(body) is not None:
                continue
            if _is_text_route(body[0], body):
                continue
            try:
                decoded = dict(decode_message(body[0], xor_decode(body)))
            except Exception:
                continue
            out.append((t, "msg", decoded))
    return out


def _elect_self_id(events: list[tuple[int, str, dict]]) -> int:
    """Vote the self tank id: whose 0x3D lands near our teleport aims."""
    votes: Counter[int] = Counter()
    pending: list[tuple[int, int, int]] = []
    for t, kind, payload in events:
        if kind == "cmd" and payload.get("kind") == "teleport":
            pending.append((t, payload["x"], payload["y"]))
            continue
        if kind != "msg" or payload.get("msg_type") not in (0x28, 0x3D):
            continue
        x, y, tank_id = payload.get("x"), payload.get("y"), payload.get("tank_id")
        if not (isinstance(x, int) and isinstance(y, int) and isinstance(tank_id, int)):
            continue
        pending = [p for p in pending if t - p[0] <= LANDING_WINDOW_MS]
        for pt, px, py in pending:
            if max(abs(x - px), abs(y - py)) <= LANDING_MAX_CHEBYSHEV:
                votes[tank_id] += 1
                break
    return votes.most_common(1)[0][0] if votes else -1


def mine(path: Path, agg: dict) -> None:
    session = json.loads(path.read_text(encoding="utf-8"))
    events = _decode_all(session)
    self_id = _elect_self_id(events)
    if self_id == -1:
        agg["sessions_no_self"] += 1
        return

    # Per-tile mine statement timeline: list of (t, present, team | -1).
    mine_timeline: dict[tuple[int, int], list[tuple[int, bool, int]]] = defaultdict(list)
    # Self team from 0x21 TankInfo (-1 until seen).
    self_team = -1
    # Last known position + time per tank.
    tank_pos: dict[int, tuple[int, int, int]] = {}
    # (t, rx, ry) awaiting landing.
    pending: list[tuple[int, int, int]] = []
    pairs: list[tuple[int, int, int, int, int]] = []  # (t, rx, ry, lx, ly)

    for t, kind, payload in events:
        if kind == "cmd":
            if payload.get("kind") == "teleport":
                pending.append((t, payload["x"], payload["y"]))
            continue
        mt = payload.get("msg_type")
        if mt in (0x28, 0x3D):
            x, y, tank_id = payload.get("x"), payload.get("y"), payload.get("tank_id")
            if not (isinstance(x, int) and isinstance(y, int) and isinstance(tank_id, int)):
                continue
            tank_pos[tank_id] = (x, y, t)
            if tank_id != self_id:
                continue
            pending = [p for p in pending if t - p[0] <= LANDING_WINDOW_MS]
            for i, (pt, px, py) in enumerate(pending):
                if max(abs(x - px), abs(y - py)) <= LANDING_MAX_CHEBYSHEV:
                    pairs.append((t, px, py, x, y))
                    pending.pop(i)
                    break
        elif mt == 0x21:
            if payload.get("tank_id") == self_id and isinstance(payload.get("team"), int):
                self_team = payload["team"]
        elif mt == 0x4F:
            for entry in payload.get("mines", []):
                mine_timeline[(entry["x"], entry["y"])].append((t, True, entry.get("team", -1)))
            for entry in payload.get("mine_clears", []):
                mine_timeline[(entry["x"], entry["y"])].append((t, False, -1))
        elif mt == 0x4B:
            for x, y in payload.get("positions", []):
                mine_timeline[(x, y)].append((t, True, -1))
        elif mt == 0x45:
            for x, y in payload.get("positions", []):
                mine_timeline[(x, y)].append((t, False, -1))
        elif mt == 0x40:
            for x, y, overlay_value in payload.get("updates", []):
                team = overlay_value & 3 if overlay_value <= 7 else -1
                mine_timeline[(x, y)].append((t, overlay_value <= 7, team))
        elif mt == 0x5A:
            left = payload.get("viewport_left")
            top = payload.get("viewport_top")
            if not (isinstance(left, int) and isinstance(top, int)):
                continue
            for entity in payload.get("entities", []):
                overlay = entity.get("overlay_value")
                if not isinstance(overlay, int):
                    continue
                tile = (left + entity["col"], top + entity["row"])
                team = overlay & 3 if overlay <= 7 else -1
                mine_timeline[tile].append((t, overlay <= 7, team))

    landings_by_request: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for t, rx, ry, lx, ly in pairs:
        displaced = (rx, ry) != (lx, ly)
        timeline = mine_timeline.get((rx, ry), [])
        before = [(present, team) for st, present, team in timeline if st <= t]
        after_reveal = any(present for st, present, _team in timeline if st > t)
        known_before = bool(before) and before[-1][0]
        mine_team = before[-1][1] if known_before else -1
        mine_is_friendly = known_before and self_team != -1 and mine_team == self_team
        body = any(
            tid != self_id and (px, py) == (rx, ry) and t - pt <= BODY_FRESH_MS
            for tid, (px, py, pt) in tank_pos.items()
        )
        if displaced:
            agg["displaced_total"] += 1
            landings_by_request[(rx, ry)].add((lx, ly))
            if known_before:
                agg["displaced_mine_known_before"] += 1
                if mine_is_friendly:
                    agg["displaced_off_friendly_mine"] += 1
                elif self_team != -1 and mine_team != -1:
                    agg["displaced_off_enemy_mine"] += 1
            elif after_reveal:
                agg["displaced_mine_revealed_after"] += 1
            elif body:
                agg["displaced_body_present"] += 1
            else:
                agg["displaced_no_visible_cause"] += 1
                distance = max(abs(lx - rx), abs(ly - ry))
                bucket = "1" if distance == 1 else ("2" if distance == 2 else "3+")
                agg["no_cause_distance"][bucket] = agg["no_cause_distance"].get(bucket, 0) + 1
                # The divergence question: had the server itself declared
                # this tile mine-free, and how recently? A fresh clean
                # statement followed by displacement is a REAL world-model
                # divergence; "never stated" is a plain sensor gap.
                clean_ages = [t - st for st, present, _team in timeline if st <= t and not present]
                if not timeline or not any(st <= t for st, _p, _tm in timeline):
                    coverage = "never_stated"
                elif not clean_ages:
                    coverage = "stated_but_never_clean"
                else:
                    age_s = min(clean_ages) / 1000
                    coverage = (
                        "clean_within_30s"
                        if age_s <= 30
                        else ("clean_within_5m" if age_s <= 300 else "clean_older")
                    )
                    if age_s <= 30 and len(agg["divergence_samples"]) < 12:
                        agg["divergence_samples"].append(
                            f"{path.name} t={t} ({rx},{ry})->({lx},{ly}) clean_{age_s:.0f}s_before"
                        )
                agg["no_cause_coverage"][coverage] = agg["no_cause_coverage"].get(coverage, 0) + 1
                # Stale-body candidate: ANY tank ever last seen on this
                # tile, no freshness cap.
                if any(
                    tid != self_id and (px, py) == (rx, ry)
                    for tid, (px, py, _pt) in tank_pos.items()
                ):
                    agg["no_cause_stale_body_candidate"] += 1
                if len(agg["no_cause_samples"]) < 10:
                    agg["no_cause_samples"].append(f"{path.name} t={t} ({rx},{ry})->({lx},{ly})")
        else:
            agg["exact_total"] += 1
            if known_before:
                agg["exact_on_known_mine"] += 1
                if mine_is_friendly:
                    agg["exact_on_friendly_mine"] += 1
                elif self_team != -1 and mine_team != -1:
                    agg["exact_on_enemy_mine"] += 1
                    if len(agg["violation_samples"]) < 12:
                        agg["violation_samples"].append(
                            f"{path.name} t={t} ({rx},{ry}) mine_team={mine_team} self={self_team}"
                        )
                else:
                    agg["exact_on_unknown_team_mine"] += 1

    for tile, landings in landings_by_request.items():
        if len(landings) > 1:
            agg["multi_landing_tiles"] += 1
        agg["repeat_displaced_tiles"] += 1 if len(landings) >= 1 else 0


def main() -> int:
    paths: list[Path] = []
    for arg in sys.argv[1:]:
        path = Path(arg)
        if path.is_dir():
            paths.extend(sorted(path.glob("*.capture_session.json")))
        else:
            paths.append(path)
    agg: dict = {
        "sessions": 0,
        "sessions_no_self": 0,
        "displaced_total": 0,
        "displaced_mine_known_before": 0,
        "displaced_off_friendly_mine": 0,
        "displaced_off_enemy_mine": 0,
        "no_cause_distance": {},
        "no_cause_coverage": {},
        "no_cause_stale_body_candidate": 0,
        "divergence_samples": [],
        "displaced_mine_revealed_after": 0,
        "displaced_body_present": 0,
        "displaced_no_visible_cause": 0,
        "exact_total": 0,
        "exact_on_known_mine": 0,
        "exact_on_friendly_mine": 0,
        "exact_on_enemy_mine": 0,
        "exact_on_unknown_team_mine": 0,
        "multi_landing_tiles": 0,
        "repeat_displaced_tiles": 0,
        "no_cause_samples": [],
        "violation_samples": [],
    }
    for path in paths:
        try:
            mine(path, agg)
            agg["sessions"] += 1
        except Exception as error:  # noqa: BLE001 - archive files vary; report and continue
            print(f"SKIP {path.name}: {error}")
    print(json.dumps(agg, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
