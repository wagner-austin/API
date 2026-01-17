"""Analyze WorldEntry message format to find tank positions."""

from __future__ import annotations

import base64
import json
from pathlib import Path

# Add src to path
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from tankpit_bot.capture.xor import build_xor_table, load_xor_static_key


def xor_decode(body: bytes, table: bytes) -> bytes:
    decoded = bytearray(len(body))
    for i in range(len(body)):
        if i < len(table):
            decoded[i] = body[i] ^ table[i]
        else:
            decoded[i] = body[i]
    return bytes(decoded)


def main():
    capture_path = project_root / "capture_session.json"
    with open(capture_path) as f:
        data = json.load(f)

    # Get magic from AUTH
    magic = None
    for msg in data["messages"]:
        if msg["direction"] != "sent":
            continue
        raw = base64.b64decode(msg["payload"])
        if len(raw) > 2 and raw[0] == 0x4F:
            text = raw[2:].decode("utf-8", errors="replace")
            if " " in text:
                parts = text.split(" ")
                magic = parts[-1].strip()
                print(f"Magic: {magic}")
                break

    # Build XOR table
    static_key, _ = load_xor_static_key(None)
    xor_table = build_xor_table(static_key, magic)

    # Find WorldEntry messages
    for msg in data["messages"]:
        if msg["direction"] != "received":
            continue
        raw = base64.b64decode(msg["payload"])
        if len(raw) < 500:
            continue

        msg_type = raw[2]
        if msg_type != 0x2E:
            continue

        body = raw[3:]
        decoded = xor_decode(body, xor_table)
        subtype = decoded[0]

        if subtype != 0x4C:  # 'L' = WorldEntry
            continue

        print(f"\n{'='*60}")
        print(f"WorldEntry: {len(decoded)} bytes")
        print(f"{'='*60}")

        # Print first 200 bytes in groups of 20
        print("\nHex dump (first 200 bytes):")
        for i in range(0, min(200, len(decoded)), 20):
            chunk = decoded[i:i+20]
            hex_str = " ".join(f"{b:02x}" for b in chunk)
            ascii_str = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
            print(f"  {i:4d}: {hex_str:<60} {ascii_str}")

        # Also dump bytes 630-660 to see what's before tank section
        print("\nHex dump around tank section (630-670):")
        for i in range(630, min(670, len(decoded)), 10):
            chunk = decoded[i:i+10]
            hex_str = " ".join(f"{b:02x}" for b in chunk)
            print(f"  {i:4d}: {hex_str}")

        # Look for known tank IDs
        print("\n\nSearching for tank IDs 500-535 and 822:")
        found_ids = []
        for i in range(len(decoded) - 1):
            val = decoded[i] | (decoded[i+1] << 8)
            if 500 <= val <= 535 or val == 822:
                found_ids.append((val, i))

        for val, offset in found_ids:
            ctx_start = max(0, offset - 8)
            ctx_end = min(len(decoded), offset + 10)
            context = decoded[ctx_start:ctx_end]
            hex_ctx = " ".join(f"{b:02x}" for b in context)
            marker_pos = offset - ctx_start
            print(f"  ID {val:4d} at offset {offset:4d}: {hex_ctx}")
            print(f"         {'   ' * marker_pos}^^")

        print(f"\nTotal tank IDs found: {len(found_ids)}")

        # The tank entries appear to start around offset 650 and are 5 bytes each:
        # [tank_id:2 LE] [flags:1] [x:1] [y:1]
        print("\n\nDecoding tank entries (5-byte format):")
        print("Format: [tank_id:2] [flags:1] [x:1] [y:1]")
        print("-" * 60)

        # Find the start of the tank entry section
        # Look for where tank IDs 500-535 start appearing densely
        tank_section_start = None
        for val, offset in found_ids:
            if 500 <= val <= 535:
                # Check if next entry is also a tank
                if offset + 5 < len(decoded):
                    next_id = decoded[offset + 5] | (decoded[offset + 6] << 8)
                    if 500 <= next_id <= 535 or next_id == 822:
                        tank_section_start = offset
                        break

        if tank_section_start:
            print(f"Tank section starts at offset {tank_section_start}")
            print()

            # Decode entries
            offset = tank_section_start
            entries = []
            while offset + 5 <= len(decoded):
                tank_id = decoded[offset] | (decoded[offset + 1] << 8)
                flags = decoded[offset + 2]
                x = decoded[offset + 3]
                y = decoded[offset + 4]

                # Stop if tank_id is out of expected range
                if not (500 <= tank_id <= 900):
                    break

                entries.append((tank_id, flags, x, y))
                offset += 5

            print(f"Found {len(entries)} tank entries:\n")
            print(f"{'ID':>5} {'Name':<12} {'Flags':>5} {'X':>4} {'Y':>4}")
            print("-" * 40)

            # Tank names
            names = {i: f"red-{i-499}" for i in range(500, 509)}
            names.update({i: f"purple-{i-508}" for i in range(509, 518)})
            names.update({i: f"blue-{i-517}" for i in range(518, 527)})
            names.update({i: f"orange-{i-526}" for i in range(527, 536)})
            names[822] = "Artax"

            for tank_id, flags, x, y in sorted(entries, key=lambda e: e[0]):
                name = names.get(tank_id, "?")
                print(f"{tank_id:5d} {name:<12} 0x{flags:02x}  {x:4d} {y:4d}")

        # Store WorldEntry positions for comparison
        worldentry_positions = {tank_id: (x, y) for tank_id, flags, x, y in entries}

        # Only analyze first WorldEntry
        break

    # Now cross-reference with other message types
    print("\n" + "=" * 60)
    print("CROSS-REFERENCE: Checking other messages for positions")
    print("=" * 60)

    # Look for TankEntry (0x28), movement (0x47), position_update (0x2E 13-byte)
    other_positions: dict[int, list[tuple[str, int, int]]] = {}

    for msg in data["messages"]:
        if msg["direction"] != "received":
            continue
        raw = base64.b64decode(msg["payload"])
        if len(raw) < 3:
            continue

        msg_type = raw[2]
        body = raw[3:]
        decoded = xor_decode(body, xor_table)

        # TankEntry 0x28: [tank_id:1][x:2 LE][y:1][...][name]
        if msg_type == 0x28 and len(decoded) >= 4:
            tank_id = decoded[0]
            x = decoded[1] | (decoded[2] << 8)
            y = decoded[3]
            if tank_id not in other_positions:
                other_positions[tank_id] = []
            other_positions[tank_id].append(("TankEntry", x, y))

        # Movement 0x2E with subtype 0x47: contains start position
        if msg_type == 0x2E and len(decoded) >= 12 and decoded[0] == 0x47:
            packed = decoded[2] | (decoded[3] << 8)
            start_x = packed >> 8
            start_y = decoded[4]
            player_id = int.from_bytes(decoded[8:12], "little")
            # player_id isn't tank_id directly, but log it
            print(f"  Movement: player_id={player_id} start=({start_x}, {start_y})")

        # Position update 0x2E 12-byte body (13 total with msg_type)
        if msg_type == 0x2E and len(decoded) == 12:
            tank_id = decoded[2] | (decoded[3] << 8)
            x = decoded[4]
            y = decoded[5]
            if 500 <= tank_id <= 900:
                if tank_id not in other_positions:
                    other_positions[tank_id] = []
                other_positions[tank_id].append(("position_update", x, y))

    print(f"\nFound positions from other messages for {len(other_positions)} tanks")

    # Compare with WorldEntry
    print("\nComparison with WorldEntry positions:")
    print("-" * 60)
    mismatches = 0
    for tank_id in sorted(worldentry_positions.keys()):
        we_x, we_y = worldentry_positions[tank_id]
        name = names.get(tank_id, "?")
        if tank_id in other_positions:
            for msg_type, x, y in other_positions[tank_id]:
                match = "MATCH" if (x == we_x and y == we_y) else "MISMATCH"
                if match == "MISMATCH":
                    mismatches += 1
                print(f"  {tank_id} {name}: WorldEntry=({we_x},{we_y}) {msg_type}=({x},{y}) {match}")
        else:
            print(f"  {tank_id} {name}: WorldEntry=({we_x},{we_y}) - no other data")

    print(f"\nTotal mismatches: {mismatches}")


if __name__ == "__main__":
    main()
