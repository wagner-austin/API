"""Inspect captured websocket data for shooting messages.

This script decodes all messages from capture_session.json and filters
for ShootEvent (0x53) and SHOOT commands to analyze homing shot behavior.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

# Add src to path so we can import tankpit_bot
project_root = Path(__file__).parent.parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

from tankpit_bot.capture.xor import build_xor_table, load_xor_static_key
from tankpit_bot.protocol.commands import (
    CMD_SHOOT,
    COMMAND_PREFIX,
    TYPE_COMBAT,
)


def load_capture_session(path: Path) -> dict:
    """Load capture session from JSON file."""
    with open(path) as f:
        return json.load(f)


def xor_decode(body: bytes, xor_table: bytes) -> bytes:
    """XOR decode body using table."""
    decoded = bytearray(len(body))
    for i in range(len(decoded)):
        if i < len(xor_table):
            decoded[i] = body[i] ^ xor_table[i]
        else:
            decoded[i] = body[i]
    return bytes(decoded)


def decode_sent_command(raw: bytes, xor_table: bytes | None) -> dict | None:
    """Decode a sent command.

    Sent commands format: [len_lo, len_hi] + body
    Body format: '!' + XOR(type_byte) + XOR(cmd_id) + XOR([payload])
    """
    if len(raw) < 4:
        return None

    # Length header (little-endian)
    length = raw[0] | (raw[1] << 8)
    body = raw[2:]

    if len(body) < 3:
        return None

    # Check for command prefix
    if body[0] != COMMAND_PREFIX:  # '!'
        return None

    # The bytes after '!' are XOR encoded
    encoded = body[1:]

    # If we have XOR table, decode the command
    if xor_table and len(encoded) > 0:
        decoded = xor_decode(encoded, xor_table)
        type_byte = decoded[0]
        cmd_id = decoded[1] if len(decoded) > 1 else 0
        payload = decoded[2:] if len(decoded) > 2 else b""
    else:
        # Without XOR table, just use raw bytes
        type_byte = encoded[0]
        cmd_id = encoded[1] if len(encoded) > 1 else 0
        payload = encoded[2:] if len(encoded) > 2 else b""

    return {
        "length": length,
        "type_byte": type_byte,
        "cmd_id": cmd_id,
        "payload": payload,
        "raw_hex": raw.hex(),
        "encoded_hex": encoded.hex(),
    }


def decode_shoot_event(data: bytes) -> dict:
    """Decode ShootEvent from XOR-decoded body."""
    if len(data) < 12:
        return {"error": f"Too short: {len(data)} bytes"}

    def x16(lo: int, hi: int) -> int:
        return lo | (hi << 8)

    def x24(b0: int, b1: int, b2: int) -> int:
        return b0 | (b1 << 8) | (b2 << 16)

    return {
        "msg_type": 0x53,
        "shooter_id": x16(data[0], data[1]),
        "target_x": data[2],
        "target_y": data[3],
        "projectile_x": data[4],
        "projectile_y": data[5],
        "fuel": x24(data[6], data[7], data[8]),
        "weapon": data[9],
        "ammo": data[10],
        "friendly_fire": data[11] == 1,
    }


def main() -> int:
    """Inspect capture session for shooting messages."""
    capture_path = project_root / "capture_session.json"
    if not capture_path.exists():
        print(f"Capture file not found: {capture_path}")
        return 1

    session = load_capture_session(capture_path)
    messages = session.get("messages", [])

    print(f"Loaded {len(messages)} messages from capture session")
    print(f"Session ID: {session.get('session_id', 'unknown')}")
    print()

    # Load XOR key
    static_key, _ = load_xor_static_key(None)
    if static_key is None:
        print("Warning: Could not load XOR static key")
        xor_table = None
    else:
        print(f"Loaded static key: {static_key[:20]}...")
        # We need magic to build the XOR table - extract from session
        magic = None

        # Look for magic in AUTH message (sent)
        # AUTH format: O + length + "AUTH !be USERID|HASH|TIMESTAMP MAGIC"
        for msg in messages:
            if msg["direction"] != "sent":
                continue
            raw = base64.b64decode(msg["payload"])
            if len(raw) > 2 and raw[0] == 0x4F:  # 'O' = AUTH
                text = raw[2:].decode("utf-8", errors="replace")
                # Magic is at the end after a space
                if " " in text:
                    parts = text.split(" ")
                    if len(parts) >= 2:
                        # Last part is magic
                        potential_magic = parts[-1].strip()
                        if len(potential_magic) >= 10 and potential_magic.isalnum():
                            magic = potential_magic
                            print(f"Found magic in AUTH: {magic}")
                            break

        if magic:
            xor_table = build_xor_table(static_key, magic)
            print(f"Built XOR table ({len(xor_table)} bytes)")
        else:
            print("Could not find magic key, trying without XOR decode")
            xor_table = None

    print()
    print("=" * 60)
    print("SENT COMMANDS ANALYSIS")
    print("=" * 60)
    print()

    # First pass: analyze sent command structure
    sent_commands: list[dict] = []
    for i, msg in enumerate(messages):
        if msg["direction"] != "sent":
            continue
        raw = base64.b64decode(msg["payload"])
        cmd = decode_sent_command(raw, xor_table)
        if cmd:
            cmd["index"] = i
            cmd["timestamp_ms"] = msg.get("timestamp_ms", 0)
            sent_commands.append(cmd)

            type_chr = chr(cmd["type_byte"]) if 32 <= cmd["type_byte"] < 127 else "."
            cmd_chr = chr(cmd["cmd_id"]) if 32 <= cmd["cmd_id"] < 127 else "."
            payload_hex = cmd["payload"].hex() if cmd["payload"] else "(none)"

            print(f"[{i:4d}] SENT type=0x{cmd['type_byte']:02X}'{type_chr}' cmd=0x{cmd['cmd_id']:02X}'{cmd_chr}' payload={payload_hex}")

            # Check if this looks like a SHOOT command
            # SHOOT uses TYPE_COMBAT=0x26 and CMD_SHOOT=0x73
            if cmd["cmd_id"] == CMD_SHOOT:
                print(f"       ^^^ SHOOT COMMAND DETECTED!")
                if len(cmd["payload"]) >= 4:
                    x, y = cmd["payload"][0], cmd["payload"][1]
                    target_id = cmd["payload"][2] | (cmd["payload"][3] << 8)
                    print(f"       target=({x},{y}) target_id={target_id}")

    print()
    print("=" * 60)
    print("RECEIVED MESSAGES - LOOKING FOR 0x53 (ShootEvent)")
    print("=" * 60)
    print()

    # MSG_SHOOT = ord("S") = 0x53
    MSG_SHOOT = 0x53

    shoot_events: list[dict] = []
    for i, msg in enumerate(messages):
        if msg["direction"] != "received":
            continue
        raw = base64.b64decode(msg["payload"])
        if len(raw) < 3:  # Need at least length(2) + msg_type(1)
            continue

        # Received messages have format: [len_lo, len_hi] + [msg_type] + [body...]
        length = raw[0] | (raw[1] << 8)
        msg_type = raw[2]
        body = raw[3:]

        # ShootEvent is 0x53 ('S')
        if msg_type == MSG_SHOOT:
            # XOR decode the body
            if xor_table:
                decoded = xor_decode(body, xor_table)
            else:
                decoded = body
            event = decode_shoot_event(decoded)
            event["index"] = i
            event["raw_hex"] = raw.hex()
            shoot_events.append(event)

            print(f"[{i:4d}] RECEIVED ShootEvent (0x53 'S')")
            print(f"       shooter_id={event.get('shooter_id')} target=({event.get('target_x')},{event.get('target_y')})")
            print(f"       projectile=({event.get('projectile_x')},{event.get('projectile_y')}) weapon={event.get('weapon')} ammo={event.get('ammo')}")
            print()

    if not shoot_events:
        print("No ShootEvent (0x53 'S') messages found in received data.")
        print()

    print()
    print("=" * 60)
    print("ALL RECEIVED MESSAGE TYPES (after length header)")
    print("=" * 60)
    print()

    msg_types: dict[int, int] = {}
    for msg in messages:
        if msg["direction"] != "received":
            continue
        raw = base64.b64decode(msg["payload"])
        if len(raw) < 3:
            continue
        msg_type = raw[2]  # Actual message type after length header
        msg_types[msg_type] = msg_types.get(msg_type, 0) + 1

    for mt in sorted(msg_types.keys()):
        count = msg_types[mt]
        char_repr = chr(mt) if 32 <= mt < 127 else "."
        print(f"  0x{mt:02X} '{char_repr}': {count}")

    # List known message types for reference
    print()
    print("Known message types:")
    print("  0x2E '.' = MSG_TANK_STATS")
    print("  0x21 '!' = MSG_TANK_INFO")
    print("  0x47 'G' = MSG_MOVEMENT")
    print("  0x53 'S' = MSG_SHOOT")
    print("  0x41 'A' = MSG_DEACTIVATE")
    print("  0x28 '(' = MSG_TANK_ENTRY")
    print("  0x29 ')' = MSG_TANK_EXIT")
    print("  0x46 'F' = MSG_RADAR_RESULT")
    print("  0x43 'C' = MSG_CONTAINER")
    print("  0x3D '=' = MSG_MOVE_RESPONSE")

    print()
    print("=" * 60)
    print("MESSAGES AROUND SHOOT COMMANDS")
    print("=" * 60)
    print()

    # Find shoot command indices
    shoot_indices = [cmd["index"] for cmd in sent_commands if cmd["cmd_id"] == CMD_SHOOT]

    for shoot_idx in shoot_indices[:3]:  # Just show first 3 for brevity
        print(f"=== SHOOT at index {shoot_idx} ===")
        # Show 5 messages before and after
        for i in range(max(0, shoot_idx - 3), min(len(messages), shoot_idx + 5)):
            msg = messages[i]
            direction = msg["direction"]
            raw = base64.b64decode(msg["payload"])

            if direction == "sent":
                cmd = decode_sent_command(raw, xor_table)
                if cmd:
                    marker = ">>>" if i == shoot_idx else "   "
                    cmd_chr = chr(cmd["cmd_id"]) if 32 <= cmd["cmd_id"] < 127 else "."
                    print(f"{marker} [{i:4d}] SENT cmd=0x{cmd['cmd_id']:02X}'{cmd_chr}'")
            else:
                if len(raw) >= 3:
                    length = raw[0] | (raw[1] << 8)
                    msg_type = raw[2]
                    body = raw[3:]

                    # XOR decode body
                    if xor_table:
                        decoded = xor_decode(body, xor_table)
                    else:
                        decoded = body

                    msg_chr = chr(msg_type) if 32 <= msg_type < 127 else "."
                    print(f"    [{i:4d}] RECV 0x{msg_type:02X}'{msg_chr}' decoded={decoded[:16].hex()}")
        print()

    print()
    print("=" * 60)
    print("TANK ID TO NAME MAPPING (from initial load)")
    print("=" * 60)
    print()

    # Parse tank info from 0x2E messages with subtype 0x21
    tank_registry: dict[int, str] = {}  # tank_id -> name

    for i, msg in enumerate(messages):
        if msg["direction"] != "received":
            continue
        raw = base64.b64decode(msg["payload"])
        if len(raw) < 3:
            continue

        msg_type = raw[2]
        body = raw[3:]

        if xor_table:
            decoded = xor_decode(body, xor_table)
        else:
            decoded = body

        # Tank info subtype 0x21: [0x21, team, tank_id_lo, tank_id_hi, ..., name]
        if msg_type == 0x2E and len(decoded) >= 11 and decoded[0] == 0x21:
            team = decoded[1]
            tank_id = decoded[2] | (decoded[3] << 8)
            # Name starts around byte 11
            name = decoded[11:].decode("utf-8", errors="replace").rstrip('\x00')
            if name and tank_id not in tank_registry:
                tank_registry[tank_id] = name

    print("Tank ID -> Name mapping:")
    for tid in sorted(tank_registry.keys()):
        print(f"  {tid:4d} (0x{tid:04X}) = '{tank_registry[tid]}'")

    print()
    print(f"Target tank_id 553 = '{tank_registry.get(553, 'NOT FOUND')}'")

    print()
    print("=" * 60)
    print("POSITION UPDATES - WHEN ARE THEY SENT?")
    print("=" * 60)
    print()

    # Look for position data in 0x2E messages
    # Subtype patterns:
    # - 0x3D '=' might be position update
    # - Look for x,y coordinates

    position_updates: list[dict] = []
    for i, msg in enumerate(messages):
        if msg["direction"] != "received":
            continue
        raw = base64.b64decode(msg["payload"])
        if len(raw) < 3:
            continue

        msg_type = raw[2]
        body = raw[3:]

        if xor_table:
            decoded = xor_decode(body, xor_table)
        else:
            decoded = body

        if msg_type == 0x2E and len(decoded) >= 6:
            subtype = decoded[0]
            # Position updates might have subtype 0x3D '=' or contain coordinate-like data
            # Also check for movement-like patterns

            # Subtype 0x3D seems to be position related based on earlier data
            # Format: [0x3D, flags, tank_id_lo, tank_id_hi, x, y, ...]
            if subtype == 0x3D and len(decoded) >= 8:
                tank_id = decoded[2] | (decoded[3] << 8)
                x = decoded[4]
                y = decoded[5]
                position_updates.append({
                    "index": i,
                    "tank_id": tank_id,
                    "x": x,
                    "y": y,
                    "hex": decoded.hex()[:20]
                })

    print(f"Found {len(position_updates)} position updates (subtype 0x3D)")
    for pu in position_updates[:10]:
        name = tank_registry.get(pu["tank_id"], "?")
        print(f"  [{pu['index']:4d}] tank_id={pu['tank_id']} ({name}) pos=({pu['x']},{pu['y']}) hex={pu['hex']}")

    print()
    print("=" * 60)
    print("ANALYZING WHEN ENEMY DATA IS RECEIVED")
    print("=" * 60)
    print()

    # Group messages by rough time/sequence to see patterns
    # Check which tank IDs appear in position updates vs initial load

    tanks_with_positions = set(pu["tank_id"] for pu in position_updates)
    tanks_in_registry = set(tank_registry.keys())

    print(f"Tanks in registry (from load): {len(tanks_in_registry)}")
    print(f"Tanks with position updates: {len(tanks_with_positions)}")
    print(f"Tanks with positions: {sorted(tanks_with_positions)}")
    print()

    # Check if we only get positions for nearby/visible tanks
    print("Position updates likely only sent for tanks in viewport or radar range.")
    print("Global map shows all tank IDs but NOT their positions.")

    print()
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print()
    print(f"Total sent commands: {len(sent_commands)}")
    print(f"Total ShootEvent received: {len(shoot_events)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
