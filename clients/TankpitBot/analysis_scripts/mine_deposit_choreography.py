"""Byte-mine the fuel-deposit wire choreography from the user sessions.

The 2026-06-20 user-piloted session (sniff-20260620-190228) holds five
manual deposits. For each, dump the full decoded window: the sent
deposit command's payload bytes, the self 0x2E fuel readings before
and after, the 0x64 receipt, and every container record with
coordinates and remaining volume — enough to state the deposit law
(amount, floor, container credit, broadcast visibility) from bytes.

Usage: ``python analysis_scripts/mine_deposit_choreography.py [capture ...]``
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.container.helpers import ContainerDecodeError
from tankpit_bot.protocol import decode_message, try_decode_plaintext_ack
from tankpit_bot.protocol.helpers import DecodeError
from tankpit_bot.sniffer.decoders import _is_text_route
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode

DEFAULT_CAPTURES = ("runs/sniff/sniff-20260620-190228.capture_session.json",)
WINDOW_MS = 12_000


def _iter_frames(data: bytes):
    frames = []
    offset = 0
    while offset + 2 < len(data):
        length = data[offset] | (data[offset + 1] << 8)
        offset += 2
        if length == 0 or offset + length > len(data):
            return frames
        frames.append(data[offset : offset + length])
        offset += length
    return frames


def _label(decoded: dict) -> str:
    msg_type = decoded.get("msg_type")
    keep = {
        k: v
        for k, v in decoded.items()
        if k in ("tank_id", "x", "y", "fuel", "fuel_total", "volume", "remaining_volume",
                 "pickups", "records", "is_free", "flag", "error_code", "reset_action",
                 "start_x", "start_y", "path")
    }
    name = f"{msg_type:#04x}" if isinstance(msg_type, int) else str(msg_type)
    return f"{name} {keep}" if keep else name


def mine(path: Path) -> None:
    session = json.loads(path.read_text(encoding="utf-8"))
    magic = session.get("magic")
    if magic is None:
        print(f"SKIP {path.name}: session has no XOR magic")
        return
    reset_xor_state()
    build_global_xor_table(magic)
    events: list[tuple[int, str]] = []
    deposit_times: list[int] = []
    for message in sorted(session["messages"], key=lambda m: m["timestamp_ms"]):
        t = message["timestamp_ms"]
        data = decode_base64_safe(message.get("payload", ""))
        if not data:
            continue
        if message.get("direction") == "sent":
            events.append((t, f"SENT {data.hex()}"))
            continue
        for body in _iter_frames(data):
            if not body or try_decode_plaintext_ack(body) is not None:
                continue
            if _is_text_route(body[0], body):
                continue
            try:
                decoded = decode_message(body[0], xor_decode(body))
            except (DecodeError, ContainerDecodeError):
                events.append((t, f"RECV undecodable {body[:2].hex()}"))
                continue
            events.append((t, "RECV " + _label(dict(decoded))))
            if decoded.get("msg_type") == 0x64:
                deposit_times.append(t)
    print(f"\n===== {path} — {len(deposit_times)} deposits =====")
    for deposit_t in deposit_times:
        print(f"\n--- window around deposit @ {deposit_t} ---")
        for t, line in events:
            if abs(t - deposit_t) <= WINDOW_MS:
                marker = ">>" if t == deposit_t and "0x64" in line else "  "
                print(f"{marker} {t - deposit_t:+7d}ms {line}")


def main() -> int:
    targets = [Path(arg) for arg in sys.argv[1:]] or [Path(p) for p in DEFAULT_CAPTURES]
    for path in targets:
        mine(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
