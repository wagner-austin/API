"""Check ActionDone (0x54) messages for success/failure payload bytes.

The current decoder throws away all bytes. Let's see what's actually there.
Also check for failed command patterns — what does the server send back
when a pickup fails, a move is blocked, etc.?
"""

from collections import defaultdict
from pathlib import Path

from tankpit_bot import _test_hooks
from tankpit_bot.capture.xor import decode_base64_safe
from tankpit_bot.sniffer.xor import build_global_xor_table, reset_xor_state, xor_decode
from tankpit_bot.types import decode_capture_session


def scan_session(session_path: Path) -> list[dict[str, object]]:
    """Scan for ActionDone and other response messages."""
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

        # Binary messages — XOR decode
        if msg_type == 0x2E:
            decoded = xor_decode(body)
            if len(decoded) < 1:
                continue

            # Check for tunneled ActionDone (0x54)
            if decoded[0] == 0x54:
                results.append({
                    "type": "action_done_tunneled",
                    "timestamp_ms": msg["timestamp_ms"],
                    "decoded_hex": decoded.hex(),
                    "decoded_bytes": list(decoded),
                    "length": len(decoded),
                })
        else:
            decoded = xor_decode(body)
            # Standalone ActionDone
            if len(decoded) >= 1 and decoded[0] == 0x54:
                results.append({
                    "type": "action_done_standalone",
                    "timestamp_ms": msg["timestamp_ms"],
                    "decoded_hex": decoded.hex(),
                    "decoded_bytes": list(decoded),
                    "length": len(decoded),
                })

    return results


def main() -> None:
    from platform_core.logging import setup_rich_logging
    setup_rich_logging(level="WARNING")

    bot_dir = Path("runs/bot")
    paths = sorted(bot_dir.glob("*.capture_session.json"))

    all_results: list[dict[str, object]] = []
    for path in paths:
        try:
            results = scan_session(path)
        except Exception:
            continue
        all_results.extend(results)

    print(f"Total ActionDone messages found: {len(all_results)}")
    print()

    # Group by type
    by_type: dict[str, list[dict[str, object]]] = defaultdict(list)
    for r in all_results:
        t = r["type"]
        assert isinstance(t, str)
        by_type[t].append(r)

    for t, msgs in sorted(by_type.items()):
        print(f"  {t}: {len(msgs)}")

    # Group by payload length
    print()
    print("By payload length:")
    by_length: dict[int, int] = defaultdict(int)
    for r in all_results:
        length = r["length"]
        assert isinstance(length, int)
        by_length[length] += 1
    for ln in sorted(by_length.keys()):
        print(f"  {ln} bytes: {by_length[ln]}")

    # Group by distinct decoded content
    print()
    print("By distinct payload (top 20):")
    by_payload: dict[str, int] = defaultdict(int)
    for r in all_results:
        h = r["decoded_hex"]
        assert isinstance(h, str)
        by_payload[h] += 1
    for payload, count in sorted(by_payload.items(), key=lambda x: -x[1])[:20]:
        raw_bytes = bytes.fromhex(payload)
        byte_list = list(raw_bytes)
        print(f"  {payload} bytes={byte_list}: {count}")

    # Show some examples with context (surrounding sent commands)
    print()
    print("Sample ActionDone messages with full bytes:")
    for r in all_results[:30]:
        db = r.get("decoded_bytes", [])
        t = r.get("type")
        ts = r.get("timestamp_ms")
        print(f"  ts={ts} type={t} bytes={db}")


if __name__ == "__main__":
    main()
