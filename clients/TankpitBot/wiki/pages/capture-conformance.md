---
title: Capture Conformance (every archived tick replayed through the sim)
tags: [sim, protocol, testing, fidelity]
related:
  - "[[capture-differ]]"
  - "[[serve-cadence]]"
  - "[[multiplayer-field]]"
  - "[[physics-module-roadmap]]"
  - "[[recipient-policy]]"
source_paths:
  - "src/tankpit_bot/validate/conformance.py"
  - "src/tankpit_bot/validate/conformance_wire.py"
  - "src/tankpit_bot/validate/conformance_mirror.py"
  - "src/tankpit_bot/validate/conformance_cli.py"
  - "tests/validate/test_conformance.py"
  - "tests/validate/test_conformance_wire.py"
  - "wiki/sources/conformance_baseline.json"
source_git_blobs:
  "src/tankpit_bot/validate/conformance.py": "3c1d434c53914a747076b159f525e8ab7d012778"
  "src/tankpit_bot/validate/conformance_wire.py": "44a781ce275f7d9a3566b18aecbc4f344ff47102"
  "src/tankpit_bot/validate/conformance_mirror.py": "8df0c339f77f5ee270584194d8071425f2e3a8bf"
  "src/tankpit_bot/validate/conformance_cli.py": "c4a93931a31e8733eb7589a5ee448c8fb86fe05e"
  "tests/validate/test_conformance.py": "e2aab37d4f457c2c7a70844b0e86d253ad5ad6e6"
  "tests/validate/test_conformance_wire.py": "010d2b3c1cb993da12a24e12e2f2158dce235e42"
  "wiki/sources/conformance_baseline.json": "837d26a995c4aafebdf90f9a668644f709d640dc"
provenance:
  - "Board task b008ab91 (the multiplayer track), Phase 3, 2026-10-05: tankpit-conformance over runs/bot and runs/sniff, the 586-capture archive copied from diphtheria:/mnt/archive-a/austinpc/tankpitbot-runs/runs"
  - "Board task b008ab91, 2026-10-07: every sent '!' frame of the 451 capture files under runs/bot and runs/probe in the API-tankpit-w1 worktree passed through sim.commands.decode_client_command by a one-off scan; 163,688 commands, none raised DecodeError"
fact_checked: "2026-10-05"
confidence: high
hubs: [protocol]
---

# Capture conformance: every archived tick replayed through the sim

*Phase 3 of the multiplayer track (board task `b008ab91`), 2026-10-05.*

A server other clients can join has to answer each tick the way the real
one does. The [[capture-differ]] asks that question in aggregate, by
wall-clock windows; `tankpit-conformance` asks it tick by tick, for every
command in every archived capture, and `make audit` now fails when the
answer gets worse.[^1]

## How a capture is replayed

- **Ticks, not windows.** The real server answers each 2 s tick with one
  batch ([[serve-cadence]]), which reaches the capture as messages a few
  milliseconds apart. A received run with no command inside it and no
  gap over 500 ms is one batch, and it answers the commands sent since
  the previous batch.[^2] The differ's 3 s windows let a slow answer
  spill into the next command's window; this pairing cannot spill.
- **Anchored, not free-running.** Before each tick the sim is set to
  what the real server last stated: the client's tile, fuel, rank,
  counts, toggles and window, every other tank's tile and life, and
  every container a radar or pickup record showed.[^3] So a tick
  compares the sim's laws from the real tick's starting state, and one
  early miss cannot make every later tick differ.
- **Every command is read, or the run stops.** A sent `!` frame whose
  body will not decode raises `DecodeError` out of `read_replay`. It is
  not dropped, because dropping it would replay its tick without it and
  compare that tick as if the client had sent less. On 2026-10-07 all
  163,688 commands in the 451 captures under `runs/bot` and `runs/probe`
  decoded, so no capture there stops a run.[^6]
- **Compared in the shape alphabet.** Both batches are reduced with the
  differ's own `shape_token`, so ids, positions and clocks never count.
  A tick is compared when the client sent it a command other than a
  keepalive. A command kind the sim has no law for is counted
  `unmodelled` instead.
- **The world is the recording's.** The joined room's field comes from
  the capture's own lobby frames. Its tanks and containers are seeded by
  the same ghost compiler `--ghost` runs use, now public as
  `run_boot.seed_ghost_world`.

The harness control: a capture the sim itself served replays with
every commanded tick matching.[^4]

## First run, 2026-10-05

437 captures replayed, 407 on field01 and 30 on field05 (Desert). Two
were skipped: the pre-framing `bot-20260331-230406`, which has no magic,
and a lobby-only sniff. **106,472 of 133,528 commanded ticks matched
(79.7%)**, with 1 unmodelled. field01 matched 80.1% and field05 78.3%,
so terrain is not what separates them.[^5]

| commands | matched | ticks | rate |
|---|---:|---:|---:|
| shoot | 34,591 | 38,343 | 90.2% |
| map_open | 20,056 | 21,114 | 95.0% |
| teleport | 15,612 | 18,699 | 83.5% |
| radar | 9,089 | 16,378 | 55.5% |
| pickup_equipment | 9,599 | 13,423 | 71.5% |
| pickup_fuel | 7,602 | 11,003 | 69.1% |
| scope | 6,831 | 7,406 | 92.2% |
| move | 2,169 | 5,352 | 40.5% |
| two or more commands in one tick | 48 | 872 | 5.5% |

What the divergences say:

- **Pickup records are timed, not just counted.** 12,916 of the 27,056
  divergent ticks (48%) differ only in where 0x43 records fall. The
  largest single row is a radar whose real batch ends in two pickup
  records the sim does not send (5,240 ticks). The differ read this as
  a landing's records spilling into the next window. Paired by batch,
  they land in the radar's own batch, one tick after the teleport, and
  the sim sends a landing's records in the landing tick (937 teleport
  ticks diverge the other way). The tick a server sends auto-pick
  records in is now an open law.
- **One command per tick.** The sim serves every command queued for a
  tank in one tick. The real server answers ticks holding two or more
  commands as if it served one: `map_open+teleport` is answered with
  the map alone in 199 of 202 ticks. This is the
  [[serve-cadence]] queue, which the sim does not model.
- **Receipts the sim never sends.** In 1,489 `move` ticks the real
  server sends a code-6 refusal and the sim sends nothing. The differ
  already files this as the sim having no in-transit state to re-click.
  Separately, in 2,345 `shoot` ticks the real batch is silent where the
  sim echoes the shot.

The ratchet is `wiki/sources/conformance_baseline.json`, the group rates
of this run. A sim change that raises a rate rewrites it with
`make conformance-baseline` in the same commit. One that lowers any
group's rate fails `make audit` with `CONFORMANCE_REGRESSED`.

[^1]: `src/tankpit_bot/validate/conformance_cli.py`, `run`: replays, prints, writes `runs/analysis/conformance.json` and applies `regressions` against `--baseline`. `make audit` passes `--baseline wiki/sources/conformance_baseline.json`.
[^2]: `src/tankpit_bot/validate/conformance_wire.py`, `BURST_GAP_MS` and `read_replay`.
[^3]: `src/tankpit_bot/validate/conformance_mirror.py`, `ArchiveMirror.observe` and `ArchiveMirror.anchor`.
[^4]: `tests/validate/test_conformance.py`, `test_the_sim_replays_its_own_capture_tick_for_tick`: a 20-round sim session's capture, replayed on open field01, has more than ten commanded ticks and every one matches.
[^5]: the run of `tankpit-conformance --write-baseline wiki/sources/conformance_baseline.json` at this page's commit, over `runs/bot` and `runs/sniff`; per-capture totals and every divergence with an example capture and timestamp are in its `runs/analysis/conformance.json` (gitignored).
[^6]: `src/tankpit_bot/validate/conformance_wire.py`, `_sent_command`, which has no catch; `tests/validate/test_conformance_wire.py`, `test_an_undecodable_command_stops_the_read`. The count is the 2026-10-07 scan named in this page's provenance.
