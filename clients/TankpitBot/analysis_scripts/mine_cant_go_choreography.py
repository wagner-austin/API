"""Byte-mine the code-1 (cant_go) walk choreography from live runs.

The F6 diagnosis says code 1 is NOT a refusal: the server accepts the
walk, moves the tank as far as the corridor allows, stops at the
first blocker, and reports. That was established with a
nearest-sample position measure (~2 s granularity); this is the
rigorous exact-window echo version the wiki records as owed.

For every ``rejected by server ... code=1`` line in a run's event
log, this pairs the logged target with the capture's wire facts in a
window around the rejection:

* every SELF 0x3D movement echo (start, path string, final tile) --
  the partial-walk evidence;
* the 0x52 receipt itself;
* every OTHER tank's last wire-known position at that instant, and
  its adjacency to the echo's final tile -- the stop-at-blocker
  evidence;
* whether NO echo exists in the window -- the pure-refusal subclass.

Usage: ``python analysis_scripts/mine_cant_go_choreography.py <run-stem> [...]``
where ``<run-stem>`` is e.g. ``runs/bot/bot-20260803-180918`` (the
``.capture_session.json`` and ``.events.jsonl`` suffixes are added).
"""

from __future__ import annotations

import datetime
import json
import re
import sys
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol import decode_message, try_decode_plaintext_ack
from tankpit_bot.sniffer.decoders import _is_text_route

# Migrated 2026-08-06 onto tankpit_bot.analysis.scan (the typed
# capture-scan owner) - the private load/XOR/frame-walk pipeline is
# deleted; results reproduce exactly. Pre-cipher discriminators (ack,
# text route) read frame["raw"], as the production receive path does.

WINDOW_BEFORE_MS = 6_000
WINDOW_AFTER_MS = 1_500

# Both log eras: pre-2026-08-03 "rejected by server (error_code=1)"
# and the current "rejected by server cant_go (code=1)".
_REJECT_LINE = re.compile(
    r"(?P<kind>move|collect|teleport) to \((?P<tx>\d+),(?P<ty>\d+)\) "
    r"rejected by server (?:\S+ )?\((?:error_)?code=1\)"
)


def _decode_all(path: Path) -> list[tuple[int, dict]]:
    """Decode every received frame with its capture timestamp."""
    result = scan_session(path)
    if "reason" in result:
        return []
    out: list[tuple[int, dict]] = []
    for frame in sorted(result["frames"], key=lambda f: f["timestamp_ms"]):
        if frame["direction"] != "received":
            continue
        if try_decode_plaintext_ack(frame["raw"]) is not None:
            continue
        if _is_text_route(frame["msg_type"], frame["raw"]):
            continue
        try:
            decoded = dict(decode_message(frame["msg_type"], frame["body"]))
        except Exception:
            continue
        out.append((frame["timestamp_ms"], decoded))
    return out


def _self_id(decoded: list[tuple[int, dict]]) -> int:
    """First 0x21 names the session's own tank (archive convention)."""
    for _, msg in decoded:
        if msg.get("msg_type") == 0x21 and isinstance(msg.get("tank_id"), int):
            return msg["tank_id"]
    raise SystemExit("no 0x21 in capture")


def _log_rejections(events_path: Path) -> list[tuple[int, str, int, int]]:
    """(wall_ms, kind, tx, ty) for every code-1 line in the event log."""
    rejections: list[tuple[int, str, int, int]] = []
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        match = _REJECT_LINE.search(str(event.get("message", "")))
        if match is None:
            continue
        stamp = datetime.datetime.fromisoformat(event["timestamp"])
        wall_ms = int(stamp.timestamp() * 1000)
        rejections.append(
            (wall_ms, match.group("kind"), int(match.group("tx")), int(match.group("ty")))
        )
    return rejections


def mine(stem: Path) -> None:
    decoded = _decode_all(stem.parent / f"{stem.name}.capture_session.json")
    self_id = _self_id(decoded)
    rejections = _log_rejections(stem.parent / f"{stem.name}.events.jsonl")

    print(f"\n===== {stem.name}: self={self_id}, {len(rejections)} code-1 lines =====")
    echoed = 0
    stopped_adjacent = 0
    for wall_ms, kind, tx, ty in rejections:
        echoes = [
            (t, msg)
            for t, msg in decoded
            if msg.get("msg_type") == 0x47
            and msg.get("tank_id") == self_id
            and isinstance(msg.get("path"), str)
            and wall_ms - WINDOW_BEFORE_MS <= t <= wall_ms + WINDOW_AFTER_MS
        ]
        receipts = [
            t
            for t, msg in decoded
            if msg.get("msg_type") == 0x52
            and msg.get("error_code") == 1
            and wall_ms - WINDOW_BEFORE_MS <= t <= wall_ms + WINDOW_AFTER_MS
        ]
        # Other tanks' last wire-stated position at the rejection instant.
        last_pos: dict[int, tuple[int, int]] = {}
        for t, msg in decoded:
            if t > wall_ms + WINDOW_AFTER_MS:
                break
            if (
                msg.get("msg_type") in (0x28, 0x3D)
                and isinstance(msg.get("tank_id"), int)
                and msg["tank_id"] != self_id
                and isinstance(msg.get("x"), int)
            ):
                last_pos[msg["tank_id"]] = (msg["x"], msg["y"])
        stamp = datetime.datetime.fromtimestamp(wall_ms / 1000).strftime("%H:%M:%S")
        print(f"\n-- {stamp} {kind} -> ({tx},{ty}): {len(receipts)} receipt(s) in window")
        if not echoes:
            print("   NO self movement echo in window  <-- pure-refusal subclass")
            continue
        echoed += 1
        for t, msg in echoes:
            fx, fy = msg["start_x"], msg["start_y"]
            for step in msg["path"]:
                if step == "n":
                    fy -= 1
                elif step == "s":
                    fy += 1
                elif step == "e":
                    fx += 1
                else:
                    fx -= 1
            neighbors = [
                (tid, pos)
                for tid, pos in last_pos.items()
                if abs(pos[0] - fx) + abs(pos[1] - fy) <= 2
            ]
            if any(abs(px - fx) + abs(py - fy) == 1 for _, (px, py) in neighbors):
                stopped_adjacent += 1
            print(
                f"   {t - wall_ms:+6d}ms echo start=({msg['start_x']},{msg['start_y']}) "
                f"path={msg['path']!r} final=({fx},{fy}) "
                f"target_delta=({tx - fx:+d},{ty - fy:+d}) "
                f"blockers_within_2={neighbors}"
            )
    print(
        f"\nsummary {stem.name}: {len(rejections)} code-1s, "
        f"{echoed} with echo, {len(rejections) - echoed} pure refusals, "
        f"{stopped_adjacent} echoes stopped cardinally adjacent to a known tank"
    )


def main() -> int:
    for arg in sys.argv[1:]:
        mine(Path(arg))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
