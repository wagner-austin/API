"""Attribute container refills to depositing tanks.

For every WITHIN-session refill in ``container_refills.json`` (a tile
read at volume ``pv`` and later at ``v > pv`` inside one capture), scan
that capture's tank-position record inside the ``[pt, t]`` gap and ask:
was any tank observed ON or ADJACENT to the refilled tile? The deposit
law ([[game-economy]]) places the banked fuel on a tile adjacent to
the depositing tank, so an adjacent sighting in the gap is the
attribution; the tank id then says WHO banks (self, practice bot
500-535, or another player).

Position sources: 0x3D MovementResponse (absolute), 0x47 Movement
(start tile plus every tile along the path string), and any 0x64
FuelDeposit the capture holds (self deposits — the 2026-07-06 manual
deposit experiments are in the archive and act as positive controls).
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol import decode_message, try_decode_plaintext_ack
from tankpit_bot.sniffer.decoders import _is_text_route
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode

REFILLS = Path("runs/analysis/container_refills.json")
RUN_DIRS = ("runs/bot", "runs/sniff", "runs/probe")
_STEP = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0)}


def _capture_path(stamp: str) -> Path | None:
    for run_dir in RUN_DIRS:
        path = Path(run_dir) / f"{stamp}.capture_session.json"
        if path.exists():
            return path
    return None


def _iter_frames(data: bytes):
    offset = 0
    while offset + 2 < len(data):
        length = data[offset] | (data[offset + 1] << 8)
        offset += 2
        if length == 0 or offset + length > len(data):
            return
        yield data[offset : offset + length]
        offset += length


def _positions_and_deposits(path: Path):
    """Extract (t, tank_id, x, y) sightings and 0x64 deposit events."""
    session = json.loads(path.read_text(encoding="utf-8"))
    reset_xor_state()
    build_global_xor_table(session["magic"])
    self_id = None
    sightings: list[tuple[int, int, int, int]] = []
    deposits: list[tuple[int, dict]] = []
    for message in sorted(session["messages"], key=lambda m: m["timestamp_ms"]):
        if message.get("direction") == "sent":
            continue
        data = decode_base64_safe(message.get("payload", ""))
        if not data:
            continue
        t = message["timestamp_ms"]
        for body in _iter_frames(data):
            if not body or try_decode_plaintext_ack(body) is not None:
                continue
            if _is_text_route(body[0], body):
                continue
            try:
                decoded = decode_message(body[0], xor_decode(body))
            except Exception:
                continue
            msg_type = decoded.get("msg_type")
            if msg_type == 0x21 and self_id is None:
                self_id = decoded.get("tank_id")
            elif msg_type == 0x3D:
                sightings.append((t, decoded["tank_id"], decoded["x"], decoded["y"]))
            elif msg_type == 0x47:
                x, y = decoded["start_x"], decoded["start_y"]
                sightings.append((t, decoded["tank_id"], x, y))
                for step in decoded.get("path", ""):
                    dx, dy = _STEP[step]
                    x, y = x + dx, y + dy
                    sightings.append((t, decoded["tank_id"], x, y))
            elif msg_type == 0x64:
                deposits.append((t, dict(decoded)))
    return self_id, sightings, deposits


def main() -> int:
    refills = json.loads(REFILLS.read_text(encoding="utf-8"))
    within = [r for r in refills if r["kind"] == "within"]
    print(f"refill events: {len(refills)} total, {len(within)} within-session")

    by_stamp = defaultdict(list)
    for r in within:
        by_stamp[r["stamp"]].append(r)

    attributed_self = attributed_bot = attributed_other = unattributed = 0
    missing_captures = 0
    slack_ms = 30_000  # sightings just before/after the gap still count
    examples: list[str] = []
    deposit_msgs_total = 0
    for stamp, events in sorted(by_stamp.items()):
        path = _capture_path(stamp)
        if path is None:
            missing_captures += len(events)
            continue
        self_id, sightings, deposits = _positions_and_deposits(path)
        deposit_msgs_total += len(deposits)
        for r in events:
            lo, hi = r["pt"] - slack_ms, r["t"] + slack_ms
            near = {
                tank_id
                for t, tank_id, x, y in sightings
                if lo <= t <= hi and abs(x - r["x"]) <= 1 and abs(y - r["y"]) <= 1
            }
            label = ""
            if not near:
                unattributed += 1
                label = "UNATTRIBUTED"
            elif near == {self_id}:
                attributed_self += 1
                label = f"SELF({self_id})"
            elif any(500 <= tank_id <= 535 for tank_id in near if tank_id != self_id):
                attributed_bot += 1
                label = f"PRACTICE-BOT {sorted(near)}"
            else:
                attributed_other += 1
                label = f"OTHER {sorted(near - ({self_id} if self_id else set()))}"
            if len(examples) < 25:
                examples.append(
                    f"    {stamp} tile ({r['x']},{r['y']}) {r['pv']}->{r['v']}: {label}"
                )

    print(f"missing captures: {missing_captures}")
    print(f"0x64 FuelDeposit messages seen across involved captures: {deposit_msgs_total}")
    print("\nwithin-session refill attribution (tank adjacent during the gap):")
    print(f"    self only:      {attributed_self}")
    print(f"    practice bot:   {attributed_bot}")
    print(f"    other player:   {attributed_other}")
    print(f"    nobody nearby:  {unattributed}")
    print("\nexamples:")
    for line in examples:
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
