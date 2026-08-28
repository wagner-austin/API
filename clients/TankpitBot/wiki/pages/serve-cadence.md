---
title: Serve Cadence (One Action per 2 s Beat)
tags: [combat, protocol, physics]
related:
  - "[[weapon-selection]]"
  - "[[game-economy]]"
  - "[[shoot-event-format]]"
source_paths:
  - "runs/probe"
  - "src/tankpit_bot/action_lab/cadence_probe.py"
  - "src/tankpit_bot/action_lab/weave_probe.py"
fact_checked: "2026-08-27"
confidence: high
hubs: [combat, protocol]
---

# Serve Cadence: One Action per 2-Second Beat

The server serves **one action per tank per ~2.0 s beat, on a global
grid shared by every tank in the room**. Measured live 2026-08-26/27
by two purpose-built probes; every claim below is echo-timestamped
wire evidence, not inference from totals (the totals lied twice
before the echoes settled it).

## The law

- **Serve rate:** exactly one action per 2 s. Our 0x53 shoot echoes
  land 1.998 / 2.002 / 2.001 s apart (`make cadence-probe`, capture
  `runs/probe/cadence-20260826-212748.capture_session.json`).
- **Global grid:** the enemy's serves land on the SAME even beats as
  ours (red-8's return fire interleaves at identical timestamps ±2 ms).
  It is a server combat tick, not a per-tank cooldown.
- **Queuing, not dropping:** excess dispatches queue and drain on the
  beat — a 6-shot burst at 500 ms spacing served 3 shots, the third
  landing 2 s AFTER the last click. Spamming buys delayed serves,
  nothing else. Swallowed dispatches are never billed (fuel deltas
  show −10 only for served shots).
- **Moves share the slot** (`make weave-probe`, capture
  `runs/probe/weave-20260827-003818.capture_session.json`): an
  alternating shoot/shoot+move pattern served
  `SHOT SHOT MOVE SHOT MOVE SHOT MOVE SHOT MOVE` — 12 commands in, 9
  actions out, one per beat. The June "server movement is instant"
  measurement was made with an empty action queue; in combat a move
  costs a shot ~1:1.

## Consequences for doctrine

- **Fire-rate is not a lever.** The bot's 2 s combat tick already
  fires at the cap; nobody on the field out-clicks anybody.
- **Weave-while-trading is dead.** Dodging halves incoming duals
  (90→45, [[weapon-selection]]'s pending-move downgrade) but costs
  our shot on that beat — the trade ratio is unchanged and the fight
  lengthens, which favors the refueling side. Movement is free ONLY
  on beats whose slot is not firing: escapes, refuel-under-fire,
  collection dwells.
- **Paired same-tick hits mean crossfire.** A −45 and −90 landing in
  one of our ticks is two attackers, never one fast human (this
  mis-read produced the false "humans click 2×/tick" claim, retracted
  2026-08-26).
- **Beat hygiene:** every mid-fight radar, map open, or scope shift
  spends a serve beat that could have been 90 damage.
- **Mines damage outside the beat economy** — mine hits are the only
  damage that costs the attacker no serve slot.

## Provenance

Cadence probe serve counts: 2000 ms → 6/6 · 500 ms → 3/6 · 250 ms →
2/6 (dispatched → served, ammo-ledger counted per
[[weapon-selection]] § per-shot ammo ledger, echo-verified). Weave
probe: 8 shots + 4 moves dispatched → 5 shots + 4 moves served on 9
consecutive beats. Both sessions ran as Arterial on Practice vs
red-8/red-2; wiki log entries of 2026-08-26 ("The server serves ~1
shot per second" — CORRECTED same night) and 2026-08-27 carry the
full measurement chain including the retracted first reading.
