---
title: Wire Recipient Policy
tags: [protocol, wire, per-recipient, broadcast, sim-fidelity]
related:
  - "[[decode-coverage]]"
  - "[[session-state-deglobalisation]]"
  - "[[server-push-gating]]"
  - "[[movable-blocks]]"
  - "[[mine-mechanics]]"
  - "[[equipment-system]]"
source_paths:
  - "src/tankpit_bot/sim/emissions.py"
  - "src/tankpit_bot/sim/combat_emissions.py"
  - "src/tankpit_bot/sim/server.py"
  - "runs/bot"
  - "runs/sniff"
  - "runs/bot/bot-20260826-003928.capture_session.json"
source_git_blobs:
  "src/tankpit_bot/sim/emissions.py": "1f06cc3bf4975943da0c3b6e54f51de96c2b8573"
  "src/tankpit_bot/sim/combat_emissions.py": "09b762c4cdd92d495ada7b45890f2ec920095bec"
  "src/tankpit_bot/sim/server.py": "57be451a9d0069cf9ca45b398791b0530efaa4fd"
fact_checked: "2026-09-01"
confidence: high
verified: 2026-09-01 (341-session archive sweep, zero-trigger test)
hubs: [protocol]
---

# Wire Recipient Policy

Which connections receive each server message. A single-client sim cannot
tell "broadcast to the room" from "send to this client" — both produce
identical output — so every unconditional emission is an undecided
ruling until the corpus decides it. Getting one wrong builds a server
that looks correct at one client and leaks another player's private
receipts at two.[^1]

## The test

For a family that names a tank, ask whether any capture carries it
naming a tank OTHER than that session's own. For a family that names no
tank, ask whether it ever arrives in a session where the client sent
ZERO of the command that would trigger it. Either answer establishes
broadcast; zero across the archive establishes per-recipient. The own
tank is the session's first 0x21, which the archive convention makes
the player's own identity ([[session-state-deglobalisation]]).[^1]

## Broadcast

| Msg | Family | Evidence |
|-----|--------|----------|
| 0x42 | BuildPickup | 1 session. `bot-20260826-003928`, own tank 601, received a 0x42 naming tank **709** (drop at 254,9, `obstacle_type=2`) having sent zero block commands[^2] |
| 0x4A | TerrainUpdate | **45 sessions** received one with zero own block presses[^1] |
| 0x45 | MineDetonation | 296 detonations against 23 placements[^3] |
| 0x4D | Chat | echoed to everyone including the sender[^4] |
| — | container_pickup | observers track consumption through the records[^5] |
| 0x41 | Deactivation | room-wide kill announcement[^3] |
| 0x53 | ShootEvent | room-wide shot echo[^3] |

The 0x42 ruling rests on ONE sample and stays high-confidence only
because it is a positive existence proof with a coherent payload — a
real drop by a real tank. The sample is small for a structural reason:
the bot has no block planner, so the whole archive holds 59 block
commands ([[movable-blocks]]).[^2]

## Per-recipient (actor only)

| Msg | Family | Received | Own triggers | Zero-trigger arrivals |
|-----|--------|----------|--------------|----------------------|
| 0x4F | RadarScanResult | 7,014 | 7,053 radar | **0**[^1] |
| 0x46 | RadarResult | 7,014 | 7,053 radar | **0**[^1] |
| 0x4C | MapData | 12,275 | 12,655 map_open | **0**[^1] |
| — | TeleportLanded | 10,541 | 10,683 teleport | **0**[^1] |
| 0x52 | Supervisor | — | — | the client only sees its own[^6] |
| 0x4B | MinePlacement | 23 placements, every one naming the capturing client[^3] |
| 0x67 | EquipmentGain | any 0x67 is a SELF gain in production[^7] |
| 0x44 | FuelGain | per-connection with the 0x52 close[^5] |
| 0x56 | Statistics | the asking tank's own counters[^8] |
| 0x5A | ViewportUpdate | the connection's stored window[^9] |

At 7,014-12,275 samples across 341 sessions, zero zero-trigger arrivals
is decisive: other players' scans and hops would otherwise land in
sessions where the client never triggered one.[^1]

0x3F Sync is per-recipient on structure rather than the trigger test —
it carries no tank id and is a per-connection view resync — and its
trigger set is genuinely multi-source (1,277 of 1,528 follow a
move).[^10]

## 0x74 is a join message, not a toggle response

Measured across 341 sessions: **324 receive exactly one** (1 receives
none, 16 receive 2-5); 267 of them at received-message ordinal 48, 59 at
49, 13 at 50; first arrival min 8.0 s / median 9.3 s / max 27.4 s after
session start; and 339 of 376 carry the same payload
`(False, True, False, True, True)` — armor off, dual on, missile off,
homing on, radar on. One per session at the tail of the join burst is a
handshake message delivering the tank's persisted equipment-enabled
state, not an answer to a toggle. The 35 toggle commands the bot ever
sent account for the multi-hit sessions.[^1]

Recipient-wise it is still actor-only: it carries no tank id and
describes the recipient's own loadout.

That ruling is STRUCTURAL, and the mechanical sweep cannot reach it:
`scripts/analyze_recipient_policy.py` reports 0x74 as `undetermined`
by design. Its zero-trigger test asks whether a family answers the
client's own COMMAND, and a join-burst family answers none — so 340
zero-trigger sessions say "not command-triggered", not "broadcast".
A sweep that ruled on it would contradict this page silently.[^1]

**Consequence for the sim — FIXED 2026-09-01.** Finding the 0x74 meant
measuring the whole join burst, and `SimServer.handshake()` diverged
from it five ways: no 0x74, a single 0x49 where the server sends two,
that 0x49 in the self block rather than the tail, an invented 0x44, and
a 0x3D riding every viewport-visible tank. The burst is now emitted as
measured and pinned by test; the shape and its consequences are in the
2026-09-01 `log.md` entry.[^11]

[^1]: Archive sweep 2026-09-01 over 341 of 342 capture sessions in `runs/bot` and `runs/sniff` (one carries no magic and cannot be XOR-decoded), decoded through `capture.frames.split_payload_frames` + `capture.xor.build_session_xor_table` + `protocol.try_decode_binary_message`, with client commands decoded through `sim.transport.decode_client_payload`. Re-runnable as `scripts/analyze_recipient_policy.py`.
[^2]: `runs/bot/bot-20260826-003928.capture_session.json`; decoded 0x42 body `{tank_id: 709, source_x: 253, source_y: 9, drop_x: 254, drop_y: 9, direction: 0, obstacle_type: 2, flag: 0}`. `direction=0` is a DROP and `obstacle_type=2` is placed-on-land per [[movable-blocks]].
[^3]: `src/tankpit_bot/sim/emissions.py::emit_mine_press` docstring, archive-cited.
[^4]: `src/tankpit_bot/sim/emissions.py::emit_chat` docstring; sniff-20260729-214411.
[^5]: `src/tankpit_bot/sim/emissions.py::emit_fuel_pickup_close` docstring; ~1,600 archive windows byte-mined 2026-08-01.
[^6]: `src/tankpit_bot/sim/emissions.py` module docstring.
[^7]: `src/tankpit_bot/sim/emissions.py::emit_equipment_pickup` docstring; 2,170 archive windows.
[^8]: `src/tankpit_bot/sim/server.py::_process_stateless_command`, the `statistics` branch.
[^9]: [[viewport-shift-protocol]]; the 0x5A origin is the connection's stored window.
[^10]: `src/tankpit_bot/sim/server_move.py::_process_move_command` docstring; archive 2026-08-06.
[^11]: `src/tankpit_bot/sim/server.py::handshake`, blob `57be451a9d0069cf9ca45b398791b0530efaa4fd`.
