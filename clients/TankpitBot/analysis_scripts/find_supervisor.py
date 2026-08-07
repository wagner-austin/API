"""Scan all captures for Supervisor (0x52 'R') messages.

Supervisor messages come through the text/protocol path, not through
0x2E containers. Our wire_byte_analysis only looked at containers.
This script scans the raw messages for 0x52.

Migrated 2026-08-06 onto ``tankpit_bot.analysis.scan`` (the typed
capture-scan owner). The original never split frames — it stripped
one length prefix (``data[2:]``) and treated the remainder of the
payload as a single frame, which mis-reads any multi-frame payload.
The per-frame walk is the correction — and measured on runs/bot
2026-08-06 it reproduced the old output EXACTLY (3,342 supervisors
both ways): every 0x52-leading frame in the corpus rides alone in
its payload, so the prefix-strip shortcut never actually bit here.
"""

from pathlib import Path

from tankpit_bot.analysis.scan import scan_session as scan_capture_session
from tankpit_bot.sniffer.constants import TEXT_MESSAGE_TYPES


def scan_session(session_path: Path) -> list[dict[str, object]]:
    """Scan a capture for Supervisor and Statistics messages."""
    result = scan_capture_session(session_path)
    if "reason" in result:
        return []

    results: list[dict[str, object]] = []
    for frame in result["frames"]:
        if frame["direction"] != "received":
            continue
        raw = frame["raw"]
        msg_type = frame["msg_type"]

        # Text messages (including Supervisor 0x52='R') — never ciphered,
        # so they read from the raw wire frame.
        if msg_type in TEXT_MESSAGE_TYPES:
            text = raw.decode("utf-8", errors="replace")
            if msg_type == 0x52:  # Supervisor
                results.append(
                    {
                        "type": "supervisor",
                        "timestamp_ms": frame["timestamp_ms"],
                        "raw_hex": raw.hex(),
                        "text": text,
                        "length": len(raw),
                    }
                )
            continue

        # Binary messages - the deciphered body, checked for 0x52 leads
        decoded = frame["body"]
        if len(decoded) >= 1 and decoded[0] == 0x52:
            results.append(
                {
                    "type": "supervisor_binary",
                    "timestamp_ms": frame["timestamp_ms"],
                    "raw_hex": raw.hex(),
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
                    "timestamp_ms": frame["timestamp_ms"],
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
