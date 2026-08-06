"""Enumerate every capture holding 0x64 FuelDeposit frames.

The deposit choreography has never been byte-mined because bot
sessions never deposit — but the archive holds USER-piloted sniff
sessions (the 2026-07-06 max-deposit experiments verified fuel
capacities, so the frames must exist). This scan finds every capture
with deposits and dumps each 0x64's decoded fields plus the
surrounding wire context (previous and next few frames), giving the
mining pass its raw windows.

Usage: ``python analysis_scripts/scan_deposit_recordings.py [--context]``
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

CAPTURE_DIRS = ("runs/sniff", "runs/bot", "runs/probe")


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


def scan_capture(path: Path, show_context: bool) -> int:
    session = json.loads(path.read_text(encoding="utf-8"))
    magic = session.get("magic")
    if not magic:
        return 0
    reset_xor_state()
    build_global_xor_table(magic)
    timeline: list[tuple[int, str, str]] = []
    deposit_indexes: list[int] = []
    for message in sorted(session.get("messages", []), key=lambda m: m["timestamp_ms"]):
        direction = message.get("direction", "received")
        data = decode_base64_safe(message.get("payload", ""))
        if not data:
            continue
        t = message["timestamp_ms"]
        if direction == "sent":
            timeline.append((t, "sent", data[:1].hex()))
            continue
        for body in _iter_frames(data):
            if not body or try_decode_plaintext_ack(body) is not None:
                continue
            if _is_text_route(body[0], body):
                continue
            try:
                decoded = decode_message(body[0], xor_decode(body))
            except (DecodeError, ContainerDecodeError):
                timeline.append((t, "recv", f"undecodable:{body[0]:#04x}"))
                continue
            msg_type = decoded.get("msg_type")
            label = f"{msg_type:#04x}" if isinstance(msg_type, int) else str(msg_type)
            if msg_type == 0x64:
                deposit_indexes.append(len(timeline))
                label = f"0x64 DEPOSIT {dict(decoded)}"
            timeline.append((t, "recv", label))
    if deposit_indexes:
        print(f"\n=== {path} — {len(deposit_indexes)} deposit frame(s) ===")
        for index in deposit_indexes:
            if show_context:
                lo = max(0, index - 6)
                hi = min(len(timeline), index + 7)
                for t, direction, label in timeline[lo:hi]:
                    marker = " >>" if timeline[index][0] == t and "DEPOSIT" in label else "   "
                    print(f"{marker} {t} {direction:4} {label}")
                print("   ---")
            else:
                t, _, label = timeline[index]
                print(f"    {t} {label}")
    return len(deposit_indexes)


def main() -> int:
    show_context = "--context" in sys.argv[1:]
    total = 0
    files = 0
    for directory in CAPTURE_DIRS:
        for path in sorted(Path(directory).glob("*.capture_session.json")):
            if path.name.startswith("latest"):
                continue
            try:
                count = scan_capture(path, show_context)
            except (OSError, json.JSONDecodeError) as error:
                print(f"    {path}: unreadable ({error})")
                continue
            total += count
            files += 1
    print(f"\nscanned {files} captures: {total} deposit frames")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
