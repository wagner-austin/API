"""The 1051-byte frame that crashed artax decodes under the wrap law.

Run bot/artax 2026-08-26 03:31:28, tick 1176: a busy practice room
(operator + two fleet bots + a human + 27 practice tanks) grew a
container-dense viewport patch to 1051 ciphered bytes — past the
1000-byte XOR table, and past the 931-byte maximum of the 282,783-body
archive the old length guard had mistaken for a protocol bound. The
real client wraps the key (``l[ja] ^= B[ja % pa]``, [[xor-cipher]]
tpclient.js case 46); the guard raised instead and killed the session.

The frame is preserved verbatim (payload + session magic) in
``oversize_frame_20260826.json`` and replays here through the full
production pipeline. Ground truth for the assertions: the decoded
patch enumerates 21 containers clustered on the viewport around
(39,53) — the equipment-hop landing artax was mid-teleport toward,
with the target container itself at (39,53).
"""

from __future__ import annotations

from pathlib import Path

from platform_core.json_utils import load_json_str, narrow_json_to_dict, require_str

from tankpit_bot import _test_hooks as core_hooks
from tankpit_bot.capture.xor import build_session_xor_table
from tankpit_bot.sniffer.decoders import process_received_message
from tankpit_bot.sniffer.world_service import WorldService

_FIXTURE = Path(__file__).with_name("oversize_frame_20260826.json")


def test_the_artax_crash_frame_replays_through_the_wrapped_cipher() -> None:
    """The exact crash frame decodes and ingests its viewport patch."""
    fixture = narrow_json_to_dict(load_json_str(core_hooks.read_text(_FIXTURE)))
    magic = require_str(fixture, "magic")
    payload = require_str(fixture, "payload")
    ws = WorldService()

    process_received_message(ws, payload, build_session_xor_table(magic))

    containers = ws.world_state["containers"]
    assert len(containers) == 21
    # The hop target artax was flying toward when the guard killed it.
    target = containers["39,53"]
    assert target["is_fuel"] is False
    # Two decoded volumes pinned against the ground-truth replay: a
    # wrong wrap origin would shred every byte past index 1000, and
    # these rows sit inside the wrapped tail's reach.
    assert containers["40,48"]["volume"] == 1177
    assert containers["43,48"]["volume"] == 1115
