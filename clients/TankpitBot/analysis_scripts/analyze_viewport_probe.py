"""Offline analysis of a viewport-probe capture: the acceptance boundary.

Reads a capture (argv[1]): pairs every sent move command (client
frames XOR-decode with the ``!`` prefix in the skipped-byte position)
with its accept (0x47 self path echo) or reject (0x52), tracked
against the latest self position (0x3D), the latest 0x5A window
origin, and the autoscroll acks (short 0x41) that split the session
into OFF/ON phases.
"""

import sys
from pathlib import Path

from tankpit_bot.analysis.scan import scan_session
from tankpit_bot.protocol import decode_message, try_decode_plaintext_ack
from tankpit_bot.sim.commands import decode_client_command

# Migrated 2026-08-06 onto tankpit_bot.analysis.scan (the typed
# capture-scan owner, direction-tagged frames) - the private
# load/XOR/frame-walk pipeline is deleted; results reproduce exactly.

path = Path(sys.argv[1] if len(sys.argv) > 1 else "viewport_probe.capture_session.json")
result = scan_session(path)
if "reason" in result:
    raise SystemExit(f"cannot analyze {path}: {result['reason']}")

self_id: int | None = None
position: tuple[int, int] | None = None
window: tuple[int, int] | None = None
phase = "OFF"
pending: tuple[str, int, int, tuple[int, int] | None, tuple[int, int] | None, str] | None = None
rows: list[tuple[str, int, int, tuple[int, int] | None, tuple[int, int] | None, str, str]] = []

for frame in sorted(result["frames"], key=lambda f: f["timestamp_ms"]):
    body = frame["body"]
    if frame["direction"] == "sent":
        if frame["msg_type"] != ord("!"):
            continue
        try:
            cmd = decode_client_command(body)
        except Exception:
            continue
        if cmd["kind"] == "move":
            pending = (cmd["kind"], cmd["x"], cmd["y"], position, window, phase)
        continue
    ack = try_decode_plaintext_ack(frame["raw"])
    if ack is not None:
        if ack["msg_type"] == "autoscroll_ack":
            phase = "ON" if ack["enabled"] else "OFF"
            print(f"--- autoscroll ack: enabled={ack['enabled']} -> phase {phase}")
        continue
    try:
        decoded = decode_message(frame["msg_type"], body)
    except Exception:
        continue
    msg_type = decoded["msg_type"]
    if msg_type == 0x21 and self_id is None:
        self_id = decoded["tank_id"]
    elif msg_type == 0x3D and decoded.get("tank_id") == self_id:
        position = (decoded["x"], decoded["y"])
    elif msg_type == 0x5A:
        window = (decoded["viewport_left"], decoded["viewport_top"])
    elif msg_type == 0x47 and decoded.get("tank_id") == self_id and pending is not None:
        rows.append((*pending, "ACCEPT"))
        pending = None
    elif msg_type == 0x52 and pending is not None:
        rows.append((*pending, "REJECT"))
        pending = None

print(f"paired moves: {len(rows)}")
for kind, tx, ty, pos, win, cmd_phase, outcome in rows:
    if pos is None:
        continue
    cheb = max(abs(tx - pos[0]), abs(ty - pos[1]))
    in_window = "?"
    if win is not None:
        in_window = str(win[0] <= tx < win[0] + 16 and win[1] <= ty < win[1] + 16)
    print(
        f"[{cmd_phase:3s}] {kind} to ({tx},{ty}) from {pos} window={win} "
        f"cheb={cheb:3d} in_window={in_window:5s} -> {outcome}"
    )
