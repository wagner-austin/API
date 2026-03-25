"""Inspect captured session to understand command encoding."""

import base64
import json
from pathlib import Path

with open("capture_session.json") as f:
    session = json.load(f)

magic = session.get("magic")
print(f"Magic: {magic}")
print()

# Build XOR table manually (no tankpit imports to avoid circular)
static_key_path = Path("src/tankpit_bot/protocol/static_key.txt")
if static_key_path.exists() and magic:
    static_key = static_key_path.read_text().strip()
    # Build XOR table: static_key[i] XOR magic[i % len(magic)]
    table = bytearray(len(static_key))
    for i in range(len(static_key)):
        table[i] = ord(static_key[i]) ^ ord(magic[i % len(magic)])
    xor_table = bytes(table)
    print(f"XOR table first 10 bytes: {xor_table[:10].hex()}")
    print()

    # Show what the REAL client sends (XOR encoded)
    sent = [m for m in session["messages"] if m["direction"] == "sent"]
    print(f"=== Real client sent ({len(sent)} messages) ===")
    for m in sent:
        data = base64.b64decode(m["payload"])
        body = data[2:]
        if len(body) < 3 or body[0] != 0x21:
            continue
        # XOR-decode to see the raw command
        decoded = bytearray(len(body))
        decoded[0] = body[0]  # '!' stays
        for i in range(1, len(body)):
            if i - 1 < len(xor_table):
                decoded[i] = body[i] ^ xor_table[i - 1]
            else:
                decoded[i] = body[i]
        wire = body.hex()[:30]
        dec = bytes(decoded).hex()[:30]
        t = decoded[1]
        c = decoded[2]
        print(f"  wire={wire:30s}  decoded={dec:30s}  type=0x{t:02x} cmd=0x{c:02x}")

    # Show what the BOT would send (no XOR encoding)
    print()
    print("=== What the bot sends (NO XOR) ===")
    # build_move_command: ! + 0x24 + 0x70 + x + y
    bot_radar = bytes([0x21, 0x22, 0x66])
    bot_move = bytes([0x21, 0x24, 0x70, 100, 100])
    print(f"  radar: {bot_radar.hex()}  type=0x{bot_radar[1]:02x} cmd=0x{bot_radar[2]:02x}")
    print(f"  move:  {bot_move.hex()}  type=0x{bot_move[1]:02x} cmd=0x{bot_move[2]:02x}")
else:
    print("Could not load static key or magic")
