"""Scan all captures for Supervisor (0x52 'R') messages.

Supervisor messages come through the text/protocol path, not through
0x2E containers. Our wire_byte_analysis only looked at containers.
This script scans the raw messages for 0x52.
"""

from pathlib import Path

from tankpit_bot import _test_hooks
from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.sniffer.constants import TEXT_MESSAGE_TYPES
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode
from tankpit_bot.types import decode_capture_session


def scan_session(session_path: Path) -> list[dict[str, object]]:
    """Scan a capture for Supervisor and Statistics messages."""
    session_text = _test_hooks.read_text(session_path)
    from platform_core.json_utils import load_json_str, narrow_json_to_dict

    session_json = narrow_json_to_dict(load_json_str(session_text))
    session = decode_capture_session(session_json)

    magic = session["magic"]
    if magic is None:
        return []

    reset_xor_state()
    build_global_xor_table(magic)

    results: list[dict[str, object]] = []
    for msg in session["messages"]:
        if msg["direction"] != "received":
            continue
        data = decode_base64_safe(msg["payload"])
        if data is None or len(data) < 3:
            continue
        body = data[2:]
        msg_type = body[0]

        # Text messages (including Supervisor 0x52='R')
        if msg_type in TEXT_MESSAGE_TYPES:
            text = body.decode("utf-8", errors="replace")
            if msg_type == 0x52:  # Supervisor
                results.append(
                    {
                        "type": "supervisor",
                        "timestamp_ms": msg["timestamp_ms"],
                        "raw_hex": body.hex(),
                        "text": text,
                        "length": len(body),
                    }
                )
            continue

        # Binary messages - XOR decode and check for 0x52 after decode
        decoded = xor_decode(body)
        if len(decoded) >= 1 and decoded[0] == 0x52:
            results.append(
                {
                    "type": "supervisor_binary",
                    "timestamp_ms": msg["timestamp_ms"],
                    "raw_hex": body.hex(),
                    "decoded_hex": decoded.hex(),
                    "decoded_bytes": list(decoded),
                    "length": len(decoded),
                }
            )

        # Also check for Statistics (0x56 'V') responses
        if len(decoded) >= 16 and decoded[0] == 0x56:
            destroyed = int.from_bytes(decoded[4:8], "little")
            deactivated = int.from_bytes(decoded[8:12], "little")
            score = int.from_bytes(decoded[12:16], "little")
            results.append(
                {
                    "type": "statistics",
                    "timestamp_ms": msg["timestamp_ms"],
                    "destroyed": destroyed,
                    "deactivated": deactivated,
                    "score": score,
                }
            )

    return results


def main() -> None:
    from platform_core.logging import setup_rich_logging

    setup_rich_logging(level="WARNING")

    bot_dir = Path("runs/bot")
    paths = sorted(bot_dir.glob("*.capture_session.json"))

    total_supervisor = 0
    total_statistics = 0

    for path in paths:
        try:
            results = scan_session(path)
        except Exception:
            continue

        supervisors = [r for r in results if r["type"] in ("supervisor", "supervisor_binary")]
        statistics = [r for r in results if r["type"] == "statistics"]

        if supervisors or statistics:
            print(f"\n{path.name}:")
            for s in supervisors:
                total_supervisor += 1
                if s["type"] == "supervisor":
                    print(
                        f"  SUPERVISOR (text): ts={s['timestamp_ms']} text={s['text']!r} hex={s['raw_hex']}"
                    )
                else:
                    db = s.get("decoded_bytes", [])
                    print(
                        f"  SUPERVISOR (binary): ts={s['timestamp_ms']} decoded={db} hex={s.get('decoded_hex')}"
                    )
                    if len(db) >= 3:
                        status = db[0]
                        print(
                            f"    status={status} {'PROMO_KILL!' if status == 8 else 'PROMO_ELIGIBLE' if status == 1 else f'other({status})'}"
                        )

            for s in statistics:
                total_statistics += 1
                print(
                    f"  STATISTICS: ts={s['timestamp_ms']} destroyed={s['destroyed']} deactivated={s['deactivated']} score={s['score']}"
                )

    print(f"\n{'=' * 60}")
    print(f"Total Supervisor messages found: {total_supervisor}")
    print(f"Total Statistics messages found: {total_statistics}")


if __name__ == "__main__":
    main()
