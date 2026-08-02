"""Teacher-forced server-law differ: recorded commands vs sim laws.

For every capture (real archive AND sim archive), pair each SENT
command with the self-caused messages the server answered inside the
command's window, and reduce them to a RESPONSE SHAPE — the ordered
tuple of message types drawn from the self-caused alphabet. Aggregate
per command kind and diff the two distributions:

* a shape the REAL server produces that the sim never does = a
  missing/incorrect sim law (this is exactly the class of bug the
  2026-08-01 ferry soak hit twice: the 0x3D/teleport_landed order and
  the 0x5A patch sort — both would surface here mechanically);
* a shape only the sim produces = an invented law.

Numeric law checks ride the same pass where the window is clean:

* teleport cost — floor(6 x euclid(start, landing)) vs the observed
  fuel delta;
* the window-bound acceptance law — a move/pickup target outside the
  stored 0x5A window must answer 0x52, inside must not answer
  0x52 code 0 (in-window rejects are world-state, tallied separately).

Usage: ``python diff_server_laws.py [--live-only|--sim-only]``.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.protocol import decode_message, try_decode_plaintext_ack
from tankpit_bot.sim.commands import decode_client_command
from tankpit_bot.sniffer.decoders import _is_text_route
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode

LIVE_DIRS = ("runs/bot", "runs/sniff", "runs/probe")
SIM_DIRS = ("runs/sim",)
WINDOW_MS = 3000
VIEWPORT_SPAN = 16


def _iter_frames(data: bytes):
    offset = 0
    while offset + 2 < len(data):
        length = data[offset] | (data[offset + 1] << 8)
        offset += 2
        if length == 0 or offset + length > len(data):
            return
        yield data[offset : offset + length]
        offset += length


def _shape_token(decoded, self_id):
    """Reduce one received message to a shape token, or None to skip.

    The alphabet is the SELF-CAUSED set: echoes and results a client's
    own command can draw. Broadcasts about other tanks and the
    periodic status syncs are background, not response.
    """
    mt = decoded.get("msg_type")
    if mt == 0x53 and decoded.get("shooter_id") == self_id:
        return "53self"
    if mt == 0x47 and decoded.get("tank_id") == self_id:
        return "47self"
    if mt == 0x3D and decoded.get("tank_id") == self_id:
        return "3Dself"
    if mt == "teleport_landed":
        return "landed"
    if mt == 0x52:
        return f"52c{decoded.get('error_code')}"
    if mt == 0x4F:
        return "4F"
    if mt == 0x46:
        return "46"
    if mt == 0x5A:
        return "5A"
    if mt == "container_pickup":
        return "pickup"
    if mt == 0x49:
        return "49"
    if mt == 0x44:
        return "44"
    if mt == 0x64:
        return "64"
    if mt == 0x67:
        return "67"
    if mt == 0x4C:
        return "4C"
    if mt == 0x4D:
        return "4D"
    return None


def mine_capture(path: Path, agg: dict, law: dict) -> None:
    """Diff one capture's command/response pairs into the aggregates."""
    try:
        session = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    magic = session.get("magic")
    messages = session.get("messages")
    if not magic or not messages:
        return
    reset_xor_state()
    build_global_xor_table(magic)
    self_id = None
    self_pos = None
    fuel_reads: list[tuple[int, int]] = []
    window = None
    # (t, kind, command) of pending sent commands awaiting their window
    sent: list[tuple[int, str, dict]] = []
    events: list[tuple[int, str, object]] = []  # (t, "cmd"|"msg", payload)

    for message in sorted(messages, key=lambda m: m["timestamp_ms"]):
        data = decode_base64_safe(message.get("payload", ""))
        if not data:
            continue
        t = message["timestamp_ms"]
        is_sent = message.get("direction") == "sent"
        for body in _iter_frames(data):
            if not body:
                continue
            if is_sent:
                if body[0] != 0x21:  # '!' command prefix; text sends skipped
                    continue
                try:
                    cmd = decode_client_command(xor_decode(body))
                except Exception:
                    continue
                events.append((t, "cmd", cmd))
                continue
            if try_decode_plaintext_ack(body) is not None or _is_text_route(body[0], body):
                continue
            try:
                decoded = decode_message(body[0], xor_decode(body))
            except Exception:
                continue
            events.append((t, "msg", decoded))

    # walk in order, tracking state and pairing windows. A window ends
    # at the NEXT sent command (or the wall cap): sim sessions compress
    # to sub-second wall time, so fixed-time windows absorb the whole
    # session there, and live deferred teleports bled into their
    # map_open's window.
    for index, (t, kind, payload) in enumerate(events):
        if kind == "cmd":
            while sent:
                _finish(sent.pop(0), agg, law, fuel_reads)
            cmd = payload
            record = {"t": t, "cmd": cmd, "shape": [], "window": window, "pos": self_pos}
            sent.append((t, cmd["kind"], record))
            agg["commands"][cmd["kind"]] += 1
            continue
        decoded = payload
        mt = decoded.get("msg_type")
        if mt == 0x21 and self_id is None:
            self_id = decoded.get("tank_id")
        token = _shape_token(decoded, self_id)
        # close expired windows
        while sent and t - sent[0][0] > WINDOW_MS:
            _finish(sent.pop(0), agg, law, fuel_reads)
        if token is not None:
            for _st, _kind, record in sent:
                record["shape"].append(token)
        if mt == 0x3D and decoded.get("tank_id") == self_id:
            self_pos = (decoded["x"], decoded["y"])
        elif mt == 0x47 and decoded.get("tank_id") == self_id:
            x, y = decoded["start_x"], decoded["start_y"]
            for step in decoded.get("path", ""):
                dx, dy = {"n": (0, -1), "s": (0, 1), "e": (1, 0), "w": (-1, 0)}[step]
                x, y = x + dx, y + dy
            self_pos = (x, y)
        elif mt == 0x5A:
            window = (decoded["viewport_left"], decoded["viewport_top"])
        elif mt == 0x44:
            fuel_reads.append((t, decoded["fuel_total"]))
        elif mt == 0x2E and decoded.get("subtype") == 2 and decoded.get("tank_id") == self_id:
            fuel = decoded.get("fuel")
            if isinstance(fuel, int):
                fuel_reads.append((t, fuel))
    while sent:
        _finish(sent.pop(0), agg, law, fuel_reads)


def _finish(entry, agg, law, fuel_reads) -> None:
    """Close one command window: record its shape and law checks."""
    _t, kind, record = entry
    shape = tuple(record["shape"])
    agg["shapes"][kind][shape] += 1
    cmd = record["cmd"]
    if kind in ("move", "pickup_fuel", "pickup_equipment") and record["window"] is not None:
        left, top = record["window"]
        inside = left <= cmd["x"] < left + VIEWPORT_SPAN and top <= cmd["y"] < top + VIEWPORT_SPAN
        rejected = any(token.startswith("52") for token in shape)
        moved = "47self" in shape
        if not inside:
            law["outside_total"] += 1
            if moved and not rejected:
                law["outside_accepted"] += 1  # LAW violation if it ever fires
        else:
            law["inside_total"] += 1
            if rejected and "52c0" in shape and not moved:
                law["inside_cant_do"] += 1  # world-state rejects, tallied
    if kind == "teleport" and record["pos"] is not None:
        # cost check needs the landing and a clean fuel bracket
        t0 = record["t"]
        before = [f for ft, f in fuel_reads if ft <= t0]
        after = [(ft, f) for ft, f in fuel_reads if t0 < ft <= t0 + WINDOW_MS]
        if before and after and "landed" in shape and "3Dself" in shape:
            law["tp_total"] += 1
            spent = before[-1] - after[0][1]
            sx, sy = record["pos"]
            expected = math.floor(
                6 * math.dist((sx, sy), (cmd["x"], cmd["y"]))
            )
            # target-based estimate; displacement makes small deviations
            if abs(spent - expected) <= 36:
                law["tp_cost_near"] += 1
            elif spent >= 0:
                law["tp_cost_far"] += 1


def sweep(dirs, prefix: str = "") -> tuple[dict, dict]:
    agg = {"commands": Counter(), "shapes": defaultdict(Counter)}
    law = defaultdict(int)
    paths = []
    for run_dir in dirs:
        paths.extend(
            path
            for path in sorted(Path(run_dir).glob("*.capture_session.json"))
            if path.name.startswith(prefix)
        )
    for index, path in enumerate(paths):
        mine_capture(path, agg, law)
        if index % 50 == 0:
            sys.stdout.write(f"  [{index + 1}/{len(paths)}] {path.name}\n")
            sys.stdout.flush()
    return agg, law


def report(name: str, agg: dict, law: dict) -> dict:
    print(f"\n=== {name} ===")
    print("commands:", dict(agg["commands"].most_common()))
    print(
        "window law: outside sent={} accepted-anyway={}  inside sent={} cant-do={}".format(
            law["outside_total"], law["outside_accepted"], law["inside_total"], law["inside_cant_do"]
        )
    )
    print(
        "teleport cost: checked={} near-law={} far={}".format(
            law["tp_total"], law["tp_cost_near"], law["tp_cost_far"]
        )
    )
    shapes = {}
    for kind, counter in sorted(agg["shapes"].items()):
        total = sum(counter.values())
        top = counter.most_common(6)
        shapes[kind] = counter
        print(f"  {kind} (n={total}):")
        for shape, count in top:
            print(f"      {100.0 * count / total:5.1f}%  {'+'.join(shape) or '(silent)'}")
    return shapes


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    sim_prefix = sys.argv[2] if len(sys.argv) > 2 else "sim-"
    live_shapes = sim_shapes = None
    if mode != "--sim-only":
        agg, law = sweep(LIVE_DIRS)
        live_shapes = report("LIVE archive", agg, law)
    if mode != "--live-only":
        agg, law = sweep(SIM_DIRS, prefix=sim_prefix)
        sim_shapes = report(f"SIM archive ({sim_prefix}*)", agg, law)
    if live_shapes is not None and sim_shapes is not None:
        print("\n=== DIVERGENCE: live shapes the sim never produces (top by count) ===")
        for kind, counter in sorted(live_shapes.items()):
            sim_seen = set(sim_shapes.get(kind, Counter()))
            missing = [(s, c) for s, c in counter.most_common() if s not in sim_seen]
            shown = [(s, c) for s, c in missing[:5] if c >= 5]
            if shown:
                print(f"  {kind}:")
                for shape, count in shown:
                    print(f"      x{count}  {'+'.join(shape) or '(silent)'}")
        print("\n=== DIVERGENCE: sim shapes never seen live ===")
        for kind, counter in sorted(sim_shapes.items()):
            live_seen = set(live_shapes.get(kind, Counter()))
            invented = [(s, c) for s, c in counter.most_common() if s not in live_seen]
            shown = [(s, c) for s, c in invented[:5] if c >= 5]
            if shown:
                print(f"  {kind}:")
                for shape, count in shown:
                    print(f"      x{count}  {'+'.join(shape) or '(silent)'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
